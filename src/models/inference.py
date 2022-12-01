from vgg_net import VGG_Net
import torch
import torchaudio
import os

CLASS_MAPPING = ["blues", "classical", "country", "disco",
                 "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

NUM_SAMPLES = 44100 * 30  # 30 seconds (1323000 Samples)
SAMPLE_RATE = 44100  # 44.1 kHz
N_FFT = 1024  # Number of samples per frame
HOP_LENGTH = 512  # Number of samples between successive frames
N_MELS = 64  # Number of mel bands

# Model Path to best saved model in saved_models directory
# TODO: Change this to your model path
MODEL_FILENAME = "VGG_20221119-140809_Tesla V100-SXM3-32GB.pt"
MODEL_PATH = "/home/zalasyu/Documents/467-CS/MGR/src/models/" + MODEL_FILENAME
print(MODEL_PATH)


class TransformInputSong:
    def __init__(self, num_samples, sample_rate, n_fft, hop_length, n_mels):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    # Makes the class callable
    # object x of class X can now be called as a function like so: x()
    def __call__(self, input):
        waveform, sample_rate = torchaudio.load(input)
        print(waveform.shape)
        waveform = self._resample(waveform, sample_rate)
        waveform = self._mixdown_to_mono(waveform)
        waveform = self._cut(waveform)
        waveform = self._right_pad(waveform)

        mel_generator = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        mel_spectrogram = mel_generator(waveform)

        # Add batch dimension
        mel_spectrogram = mel_spectrogram.unsqueeze(0)

        return mel_spectrogram

    def _mixdown_to_mono(self, signal):
        """
        Mixes down the signal to mono
        Args:
            signal (Tensor): audio signal
        Returns:
            signal (Tensor): audio signal
        """
        # If the signal is stereo, convert it to mono
        if signal.shape[0] > 1:
            # Mean the signal across the channels
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut(self, signal):
        """
        Cuts the signal to the specified number of samples
        Args:
            signal (Tensor): audio signal
        Returns:
            signal (Tensor): audio signal
        """
        signal = signal[:, :self.num_samples]
        return signal

    def _right_pad(self, signal):
        """
        Pads the signal to the specified number of samples
        Args:
            signal (Tensor): audio signal
        Returns:
            signal (Tensor): audio signal
        """
        signal = torch.nn.functional.pad(
            signal, (0, self.num_samples - signal.shape[1]))
        return signal

    def _resample(self, signal, sr):
        """
        Resamples the signal to the specified sample rate
        Args:
            signal (Tensor): audio signal
            sr (int): sample rate
        Returns:
            signal (Tensor): audio signal
        """
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate)
            signal = resampler(signal)
        return signal


class Oracle():
    """
    Get predictions from a model with a given input
    """

    def __init__(self):
        self.model = VGG_Net()
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        self.transform_input_song = TransformInputSong(
            NUM_SAMPLES, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS)

    def get_predictions(self, input):
        """
        Predicts the genre of a song
        Args:
            mel_spectrogram (Tensor): mel spectrogram of input song
            model (nn.Module): model to use for prediction
        """
        outputs = self.model(self.transform_input_song(input))
        _, predicted = torch.max(outputs.data, 1)
        # Show Confidence for each class
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        print(f"Confidence: {confidence.tolist()}")

        mapped = self._map_class_to_confidence(confidence.tolist())

        return mapped

    def _map_class_to_confidence(self, confidence_list):
        """
        Maps the class index to the class name and returns the confidence
        Args:
            confidence_list (list): list of confidences (Each index corresponds to a class)
        """
        # Combine corresponding class and confidence into a dictionary
        mapped_class = dict(zip(CLASS_MAPPING, confidence_list))
        return mapped_class


class InputChecker():

    def __init__(self):
        self.valid_extensions = [".mp3", ".wav"]

    def __call__(self, input):
        self.check_input(input)
        self.check_extension(input)
        self.check_file_size(input)

    def check_input(self, input):
        if not os.path.exists(input):
            raise ValueError("Input file does not exist")

    def check_extension(self, input):
        if os.path.splitext(input)[1] not in self.valid_extensions:
            raise ValueError("Input file is not a valid audio file")

    def check_file_size(self, input):
        if os.path.getsize(input) == 0:
            raise ValueError("Input file is empty")


if __name__ == "__main__":
    model = VGG_Net()
    transformer = TransformInputSong(
        num_samples=NUM_SAMPLES,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    oracle = Oracle()
    input_checker = InputChecker()

    try:
        # 1. Get input from user
        userInput = input("Enter the path to the song: ")

        # 2. Check input
        input_checker(userInput)

        # 3. Get predictions
        oracle(userInput)

    except Exception as e:
        print(e)
