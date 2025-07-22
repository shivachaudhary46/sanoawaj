import pyaudio
import wave 
import threading
import time
import os
from datetime import datetime
import numpy as np


class audio_recorder:
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        '''
            initalize the audio recorder for recording voice in a each 5 second

            Args:
            sample_rate in an (int): Audio sample rate (16000 is good for speech)
            channels is an(int): Number of audio channels (1 = mono, 2 = stereo)
        '''
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16  # 16-bit audio
        self.is_recording = False
        self.audio_data = []
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

    def continuous_5_second_recording(self, max_recordings=None, interval=0, output_dir="recordings"):
        '''
        Continuously record 5-second audio clips
        
        Args:
            max_recordings (int): Maximum number of recordings (None = infinite)
            interval (float): Pause between recordings in seconds
            output_dir (str): Directory to save recordings
        '''
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        recording_count = 0
        
        print(f"🔄 Starting continuous 5-second recording...")
        print(f"   Output directory: {output_dir}")
        print(f"   Max recordings: {max_recordings or 'Unlimited'}")
        print(f"   Interval: {interval} seconds")
        print("   Press Ctrl+C to stop")
        
        try:
            while max_recordings is None or recording_count < max_recordings:
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"audio_5sec_{timestamp}.wav")
                
                # Record 5 seconds
                result = self.record_5_second_wav(filename)
                
                if result:
                    recording_count += 1
                    print(f"📁 Recording #{recording_count} saved: {result}")
                    
                    # Wait before next recording
                    if interval > 0:
                        print(f"⏱️ Waiting {interval} seconds...")
                        time.sleep(interval)
                else:
                    print("❌ Recording failed, stopping...")
                    break
        except KeyboardInterrupt:
            print(f"\n🛑 Stopped by user. Total recordings: {recording_count}")


    def save_wav_file(self, filename):
        '''
        Save recorded audio data to a WAV file
        
        Args:
            filename (str): Output filename
        '''
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_data))
            
            # Get file size for confirmation
            file_size = os.path.getsize(filename)
            print(f"💾 Saved: {filename} ({file_size:,} bytes)")
            
        except Exception as e:
            print(f"❌ Error saving file: {e}")
    

    def record_5_second_wav(self, output_filename=None, device_index=None):
        '''
        Record exactly 5 seconds of audio and save as WAV file
        
        Args:
            output_filename (str): Output WAV filename (auto-generated if None)
            device_index (int): Audio device index (default device if None)
            
        Returns:
            str: Path to the saved WAV file
        '''
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"audio_5sec_{timestamp}.wav"
        
        print(f"🎙️ Recording 5 seconds of audio...")
        print(f"   Output: {output_filename}")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Channels: {self.channels}")
        
        try:
            # Open audio stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.audio_data = []
            self.is_recording = True
            
            # Calculate how many chunks we need for 5 seconds
            chunks_needed = int(self.sample_rate / self.chunk_size * 5)
            
            print("🔴 Recording started... Speak now!")
            
            # Record for exactly 5 seconds
            for i in range(chunks_needed):
                data = stream.read(self.chunk_size)
                self.audio_data.append(data)
                
                # Show progress
                progress = (i + 1) / chunks_needed * 100
                print(f"\r   Progress: {progress:.1f}%", end="")
            
            print(f"\n✅ Recording completed!")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            self.is_recording = False
            
            # Save to WAV file
            self.save_wav_file(output_filename)
            
            return output_filename
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
        
def main():
    recorder = audio_recorder()

    recorder.continuous_5_second_recording(
        max_recordings=10,
        interval = 5
    )

main()