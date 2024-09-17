import whisperx
import gc
import os
import json
import gzip
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.progress import track
import re
from datetime import timedelta, datetime

# Step 1
# ---------------------------------------------------------------------------- #
#                                  TRANSCRIBE                                  #
# ---------------------------------------------------------------------------- #
def transcribe_audio_file(audio_file, progress):
    """Transcribe the given audio file using the Whisper model."""
    device = "cuda" # GPU acceleration
    batch_size = 16 # GPU memory usage
    compute_type = "float16" # GPU accuracy

    # Load the Whisper model
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    # Load and process the audio file
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # Load alignment model for more precise word timings
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    # Align transcription for more accurate word timings
    aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

    return aligned_result["segments"]


# Step 2
# ---------------------------------------------------------------------------- #
#                         SPLIT SEGMENTS ON LARGE GAPS                         #
# ---------------------------------------------------------------------------- #
def split_segments_on_large_gaps(segments, time_threshold=2.0):
    """Split segments if there is a large time gap between words."""
    new_segments = []
    
    for segment in segments:
        words = segment["words"]
        if not words:
            continue
        
        current_words = [words[0]]
        current_start = words[0].get("start", 0)
        current_text = words[0]["word"]
        
        for i in range(1, len(words)):
            prev_word = words[i-1]
            current_word = words[i]
            
            # Check if both start and end times exist for both words
            if "end" in prev_word and "start" in current_word:
                # Calculate the gap between the end of the previous word and the start of the current word
                gap = current_word["start"] - prev_word["end"]
            else:
                gap = 0  # If either word is missing time values, assume no gap
            
            if gap > time_threshold:
                # If the gap is larger than the threshold, split the segment
                # Create a new segment for the current collected words
                new_segments.append({
                    "start": current_words[0].get("start", 0),
                    "end": current_words[-1].get("end", 0),
                    "text": " ".join([word["word"] for word in current_words]),
                    "words": current_words
                })
                
                # Reset current words and start collecting for the next segment
                current_words = [current_word]
                current_start = current_word.get("start", 0)
                current_text = current_word["word"]
            else:
                # Continue adding words to the current segment
                current_words.append(current_word)
                current_text += " " + current_word["word"]
        
        # Add the last collected segment
        new_segments.append({
            "start": current_words[0].get("start", 0),
            "end": current_words[-1].get("end", 0),
            "text": " ".join([word["word"] for word in current_words]),
            "words": current_words
        })

    return new_segments

# Function to save word segments to a compressed JSON file
def save_segments_to_gzip_json(transcription_segments, output_file):
    """Save word segments to a compressed JSON file using gzip."""
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(transcription_segments, f, indent=4)
    console.print(f"[green]Word segments saved to {output_file}[/green]")

# Function to load word segments from a compressed JSON file
def load_segments_from_gzip_json(input_file):
    """Load word segments from a compressed JSON file using gzip."""
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        return json.load(f)


# Step 3
# ---------------------------------------------------------------------------- #
#                              CONVERT JSON TO SRT                             #
# ---------------------------------------------------------------------------- #
# Convert seconds to SRT time format (ensure HH:MM:SS,mmm format)
def format_time(seconds):
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

# Split text based on character limit (without breaking words)
def split_text_by_length(text, words, max_length):
    lines = []
    current_line = ""
    current_words = []

    for word_data in words:
        word = word_data['word']
        if len(current_line + " " + word) <= max_length:
            current_line += (" " if current_line else "") + word
            current_words.append(word_data)
        else:
            lines.append((current_line, current_words))
            current_line = word
            current_words = [word_data]
    
    if current_line:
        lines.append((current_line, current_words))
    
    return lines

# Estimate missing word time
def estimate_missing_time(prev_word, next_word):
    return (prev_word['end'] + next_word['start']) / 2

# Convert transcription to SRT with proper time formatting
def convert_json_to_srt(transcription_segments, max_chars=30):
    srt_output = []
    index = 1

    for segment in transcription_segments:
        start_time = segment["start"]
        end_time = segment["end"]
        words = segment["words"]

        # Handle words without timestamps (e.g., "100%")
        for i, word_data in enumerate(words):
            if "start" not in word_data or "end" not in word_data:
                if i > 0 and i < len(words) - 1:
                    # Estimate missing time for words in the middle
                    word_data["start"] = estimate_missing_time(words[i-1], words[i+1])
                    word_data["end"] = estimate_missing_time(words[i-1], words[i+1])
                elif i == 0 and len(words) > 1:
                    # Handle missing time for the first word
                    word_data["start"] = words[i+1]["start"]
                    word_data["end"] = words[i+1]["start"]
                elif i == 0 and len(words) == 1:
                    # Handle case where there's only one word in the segment
                    word_data["start"] = start_time
                    word_data["end"] = end_time
                else:
                    # Handle missing time for the last word
                    word_data["start"] = words[i-1]["end"]
                    word_data["end"] = words[i-1]["end"]

        # Create the full text and split if necessary
        full_text = " ".join([word_data["word"] for word_data in words])
        split_lines = split_text_by_length(full_text, words, max_chars)

        # Adjust times for each split line
        for line_text, line_words in split_lines:
            line_start_time = line_words[0]["start"]
            line_end_time = line_words[-1]["end"]

            # Add SRT index
            srt_output.append(f"{index}")
            # Add time range (ensure proper formatting)
            srt_output.append(f"{format_time(line_start_time)} --> {format_time(line_end_time)}")
            # Add caption text
            srt_output.append(line_text)
            srt_output.append("")  # Newline for next caption

            index += 1

    return "\n".join(srt_output)

# Save to SRT file
def save_srt_file(output_file, srt_content):
    """Save the SRT content to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(srt_content)

# Load SRT file
def read_srt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# Step 4
# ---------------------------------------------------------------------------- #
#                                  PROCESS SRT                                 #
# ---------------------------------------------------------------------------- #
def srt_time_to_timedelta(time_str):
    """Convert SRT time format (00:04:03,860) to timedelta."""
    return datetime.strptime(time_str, '%H:%M:%S,%f') - datetime(1900, 1, 1)

def timedelta_to_srt_time(td):
    """Convert timedelta to SRT time format (00:04:03,860)."""
    total_seconds = int(td.total_seconds())
    milliseconds = int(td.microseconds / 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}'

def process_srt(srt_text, gap_threshold=0.5, extend_time=0.5, silence_threshold=5):
    """Process the SRT file as per the user request."""
    captions = re.split(r'\n\s*\n', srt_text.strip())  # Split on blank lines between blocks
    new_captions = []

    # Parse each caption block
    for i, caption in enumerate(captions):
        lines = caption.split('\n')
        idx = lines[0].strip()
        time_range = lines[1].strip()
        text = ' '.join(lines[2:]).strip()

        # Parse start and end times
        start_time_str, end_time_str = time_range.split(' --> ')
        start_time = srt_time_to_timedelta(start_time_str)
        end_time = srt_time_to_timedelta(end_time_str)

        # Extend end time by 0.5 seconds (unless there is a caption after)
        if i < len(captions) - 1:
            next_caption_time_str = re.split(r'\n', captions[i+1])[1].split(' --> ')[0]
            next_start_time = srt_time_to_timedelta(next_caption_time_str)

            # Check if the gap between captions is less than the threshold
            if (next_start_time - end_time).total_seconds() <= gap_threshold:
                end_time = next_start_time  # Align with the next caption's start time
            else:
                end_time += timedelta(seconds=extend_time)
        else:
            # Last caption: just extend by the given time
            end_time += timedelta(seconds=extend_time)

        # Add period if there's a gap larger than 5 seconds to the next caption
        if i < len(captions) - 1:
            next_start_time = srt_time_to_timedelta(re.split(r'\n', captions[i+1])[1].split(' --> ')[0])
            if (next_start_time - end_time).total_seconds() > silence_threshold and not text.endswith(('.', ',', '!', '?')):
                text += '.'
        else:
            # If it's the last caption, check for the gap till the end of the video
            if not text.endswith(('.', ',', '!', '?')):
                text += '.'

        # Rebuild the caption block
        new_time_range = f'{timedelta_to_srt_time(start_time)} --> {timedelta_to_srt_time(end_time)}'
        new_captions.append(f'{idx}\n{new_time_range}\n{text}')

    return '\n\n'.join(new_captions)


# Step 5
# ---------------------------------------------------------------------------- #
#                          REPLACE INAPPROPRIATE TEXT                          #
# ---------------------------------------------------------------------------- #
def replace_inappropriate_text(srt_file, output_file, replacements):
    def apply_replacements(text, replacements):
        """Replace inappropriate words in the given text based on replacements dictionary."""
        for find_text, replace_text in replacements.items():
            # Make the plural form optional (match both singular and plural)
            text = re.sub(r'\b' + re.escape(find_text) + r'(s?)\b(?=\b|[.,!?])', replace_text + r'\1', text, flags=re.IGNORECASE)
        return text

    # Read the original SRT file
    with open(srt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Apply replacements to each line
    new_lines = []
    for line in lines:
        if line.strip() and not line.strip().isdigit():  # Skip digit-only lines
            new_lines.append(apply_replacements(line, replacements))
        else:
            new_lines.append(line)
    
    # Write the modified SRT to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    console.print(f"[green]SRT with replaced inappropriate text saved to {output_file}[/green]")


# --------------------------------- CLEAN UP --------------------------------- #
def cleanup_files(*files):
    """Remove the specified files if they exist."""
    for file in files:
        if os.path.isfile(file):
            try:
                os.remove(file)
                console.print(f"[blue]Removed file: {file}[/blue]")
            except OSError as e:
                console.print(f"[red]Error removing file {file}: {e}[/red]")
        else:
            console.print(f"[yellow]File not found: {file}[/yellow]")


# ---------------------------------------------------------------------------- #
#                                   START APP                                  #
# ---------------------------------------------------------------------------- #
def main():
    """Main function to process audio files and generate SRT files."""
    input_folder = input('Enter the path to the input folder: ')

    # Initialize rich progress bar
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
        for file_name in os.listdir(input_folder):
            if file_name.endswith(('.mp3', '.wav')):
                audio_file = os.path.join(input_folder, file_name)

                # -------------------------------- TRANSCRIBE -------------------------------- #
                output_file_json = os.path.join(input_folder, f"{os.path.splitext(file_name)[0]}_1_transcription_segments.json.gz")

                # Check if output_file_json has already been transcribed
                if not os.path.isfile(output_file_json):
                    console.print(f"[cyan]Processing {audio_file}...[/cyan]")
                    task = progress.add_task(f"Transcribing {file_name}", total=None)
                    
                    # Transcribe and save to file
                    transcription_segments = transcribe_audio_file(audio_file, progress)
                    save_segments_to_gzip_json(transcription_segments, output_file_json)

                    progress.remove_task(task)
                else:
                    console.print(f"[yellow]Transcription segments already exist for {audio_file}. Skipping transcription.[/yellow]")
                    transcription_segments = load_segments_from_gzip_json(output_file_json)

                # ----------------------- SPLIT SEGMENTS ON LARGE GAPS ----------------------- #
                output_modified_file_json = os.path.join(input_folder, f"{os.path.splitext(file_name)[0]}_2_modified_segments.json.gz")

                # Check if output_file_json has already been transcribed
                if not os.path.isfile(output_modified_file_json):
                    
                    # Modify the transcription segments to split where needed
                    modified_segments = split_segments_on_large_gaps(transcription_segments)
                    save_segments_to_gzip_json(modified_segments, output_modified_file_json)

                else:
                    modified_segments = load_segments_from_gzip_json(output_modified_file_json)


                # ---------------------------- CONVERT JSON TO SRT --------------------------- #
                output_convert_file_srt = os.path.join(input_folder, f"{os.path.splitext(file_name)[0]}_3_convert.srt")
                
                # Check if output_file_srt has already been generated
                if not os.path.isfile(output_convert_file_srt):

                    # Create SRT content and save to file
                    srt_content = convert_json_to_srt(modified_segments)
                    save_srt_file(output_convert_file_srt, srt_content)
                    
                    console.print(f"[green]SRT file saved to {output_convert_file_srt}[/green]")

                else:
                    srt_content = read_srt_file(output_convert_file_srt)
            
            
                # -------------------------------- PROCESS SRT ------------------------------- #
                output_process_file_srt = os.path.join(input_folder, f"{os.path.splitext(file_name)[0]}_4_process.srt")
                
                # Check if output_file_srt has already been generated
                if not os.path.isfile(output_process_file_srt):

                    # Create SRT content and save to file
                    process_srt_content = process_srt(srt_content)
                    save_srt_file(output_process_file_srt, process_srt_content)
                    
                    console.print(f"[green]SRT file saved to {output_process_file_srt}[/green]")
            
                # ------------------------ REPLACE INAPPROPRIATE TEXT ------------------------ #
                output_file_srt = os.path.join(input_folder, f"{os.path.splitext(file_name)[0]}.srt")
                
                # Check if the SRT_replace file exists
                if not os.path.isfile(output_file_srt):
                    
                    # Load replacements from a JSON file
                    with open('app/badwords.json', 'r') as f:
                        replacements = json.load(f)
                    
                    replace_inappropriate_text(output_process_file_srt, output_file_srt, replacements)
            
                # --------------------------------- CLEAN UP --------------------------------- #
                # Call cleanup_files with the files you want to remove
                cleanup_files(output_modified_file_json, output_convert_file_srt, output_process_file_srt)
            
# Entry point of the script
if __name__ == "__main__":
    # Set up rich console for pretty printing
    console = Console()

    # Run the main function
    main()