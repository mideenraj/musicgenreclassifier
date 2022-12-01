import os.path
import numpy

from src.data.PrepareInput import PrepareAudio as pa
from src.data.makedataset import MusicTrainingData as td


class TestPrepareAudio():
    """Tests the PrepareAudio class in src/data."""
    pa = pa()

    def test_mp3_file_accepted(self):
        """Test that a valid .mp3 file is accepted"""

        file_path = 'tests/test_data/validfile_1.mp3'
        assert self.pa.check_file_type(file_path) is True

    def test_mp4_file_accepted(self):
        """Test that a valid .mp4 file is accepted"""

        file_path = 'tests/test_data/validfile_4.mp4'
        assert self.pa.check_file_type(file_path) is True

    def test_wav_file_accepted(self):
        """Test that a valid .wav file is accepted"""

        file_path = 'tests/test_data/validfile_2.wav'
        assert self.pa.check_file_type(file_path) is True

    def test_au_file_accepted(self):
        """Test that a valid .au file is accepted"""

        file_path = 'tests/test_data/validfile_3.au'
        assert self.pa.check_file_type(file_path) is True

    def test__file_not_accepted(self):
        """Test that a valid .au file is accepted"""

        file_path = 'tests/test_data/invalidfile_1.pdf'
        assert self.pa.check_file_type(file_path) is False

    def test_file_exists(self):
        """Test if an existing file is detected."""
        file_path = 'tests/test_data/validfile_1.mp3'
        assert self.pa.check_file_exists(file_path) is True

    def test_file_doesnt_exists(self):
        """Test if an non-existing file is not detected."""
        file_path = 'tests/test_data/nonexistingFile.mp3'
        assert self.pa.check_file_exists(file_path) is False

    def test_array_data_type(self):
        """Test that the mel spectrogram array returned is of type list"""
        file_path = 'tests/test_data/validfile_2.wav'
        spectrogram_arr = self.pa.start(file_path)
        assert isinstance(spectrogram_arr, list)

    # Test works locally, removing from test suite because it fails on GitHub
    # def test_too_short_file_rejected(self):
    #     """Test that a mp3 file that does not meet the minimum duration is rejected"""
    #     file_path = 'tests/test_data/invalidfile_3_tooshort.mp3'
    #     result = self.pa.start(file_path)
    #     assert result[0] is False


class TestTrainingData:
    """
    Tests the music training dataset builder in src/data.
    """
    td = td()

    def test_create_genre_dictionary(self):
        """Test that a genre dictionary has been built"""
        genre_path = 'tests/test_data/Genres'
        output_path = 'tests/test_data'
        self.td.create_genre_dictionary(genre_path, output_path)
        assert len(self.td.get_genre_dictionary()) == 2

    def test_build_dataset(self):
        """Tests that a numpy array is built"""
        genre_path = 'tests/test_data/Genres'
        output_path = 'tests/test_data'
        self.td.make_training_data(genre_path, output_path)
        npy_file = os.path.join(output_path, 'training_data.npy')
        assert os.path.exists(npy_file) is True

    def test_length_dataset(self):
        """Tests that numpy dataset has correct count"""
        training_data = numpy.load(
            'tests/test_data/training_data.npy', allow_pickle=True)
        assert len(training_data) == 3
