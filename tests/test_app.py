import unittest
import sys
import os
import io
from unittest.mock import MagicMock, patch
import numpy as np

# Add src to path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock modules BEFORE importing app
mock_tensorflow = MagicMock()
mock_keras = MagicMock()
mock_models = MagicMock()
mock_cv2 = MagicMock()
mock_pil = MagicMock()
mock_image = MagicMock()

sys.modules['tensorflow'] = mock_tensorflow
sys.modules['tensorflow.keras'] = mock_keras
sys.modules['tensorflow.keras.models'] = mock_models
sys.modules['tensorflow.keras.applications'] = MagicMock()
sys.modules['tensorflow.keras.applications.mobilenet_v2'] = MagicMock()
sys.modules['cv2'] = mock_cv2
sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_image

# Mock the load_model function to return a mock model
mock_classification_model = MagicMock()
mock_classification_model.predict.return_value = np.array([[0.1, 0.8, 0.05, 0.05]])
mock_models.load_model.return_value = mock_classification_model

# Now we can import app
import app as app_module
from app import app

class TestWheatApp(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        
        # Patch the module-level classification_model
        self.classification_model_patcher = patch.object(app_module, 'classification_model', mock_classification_model)
        self.mock_model = self.classification_model_patcher.start()
    
    def tearDown(self):
        self.classification_model_patcher.stop()



    def test_index_route(self):
        """Test that the index page loads correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Wheat Disease Detection', response.data)

    def test_predict_no_file(self):
        """Test prediction without sending a file."""
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 302)  # Redirects

    def test_predict_invalid_file(self):
        """Test prediction with a text file instead of an image."""
        data = {
            'file': (io.BytesIO(b"fake image data"), 'test.txt')
        }
        response = self.client.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 302) # Should redirect due to invalid extension

    def test_predict_valid_file(self):
        """Test prediction with a valid dummy image."""
        # Create a minimal valid JPEG
        img_bytes = io.BytesIO(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00\x00?\x00\xbf\x00')
        
        data = {
            'file': (img_bytes, 'test.jpg')
        }
        
        # Mock PIL.Image.open to return a proper mock image
        mock_img = MagicMock()
        mock_img_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Patch the functions in the app module context
        with patch('app.Image.open') as mock_pil_open, \
             patch('app.np.array') as mock_np_array, \
             patch('app.cv2.cvtColor') as mock_cvt, \
             patch('app.cv2.resize') as mock_resize, \
             patch('app.cv2.imwrite') as mock_write:
            
            mock_pil_open.return_value = mock_img
            mock_np_array.return_value = mock_img_array
            mock_cvt.return_value = mock_img_array
            mock_resize.return_value = mock_img_array
            
            response = self.client.post('/predict', data=data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Detected Condition', response.data)

if __name__ == '__main__':
    unittest.main()
