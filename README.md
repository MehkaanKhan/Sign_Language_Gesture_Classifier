# Sign Language Gesture Classifier

A deep learning project that classifies American Sign Language (ASL) gestures from images using a trained ResNet model. This project demonstrates image classification, transfer learning, and interactive inference using FastAI and Gradio.


# Project Overview

This project aims to help in recognizing ASL gestures to facilitate communication between hearing and non-hearing individuals.
Key features:
* Classifies 26 ASL alphabet gestures
* Trained on a custom ASL dataset
* Supports image input for real-time predictions
* Easy-to-use Gradio interface for demonstration


# Technologies Used

* Python 3.x
* [FastAI](https://www.fast.ai/)
* [PyTorch](https://pytorch.org/)
* Gradio [optional, for interactive demo(https://6e75a86b7724adb30c.gradio.live/)]
* Jupyter Notebook / Google Colab


# Project Structure

```
SignLanguageClassifier/
│
├── export2.pkl            # Trained model weights (large file, download separately)
├── SignLanguage.ipynb    # Jupyter notebook with training & inference code
├── README.md             # Project description
└── requirements.txt      # Python dependencies
```

# Setup Instructions

1. *Clone the repository*

```bash
git clone https://github.com/your-username/SignLanguageClassifier.git
cd SignLanguageClassifier
```

2. *Install dependencies*

```bash
pip install -r requirements.txt
```

3. *Download the trained model*

* Model file (`export2.pkl`) is hosted on [Hugging Face](https://huggingface.co/Mehkaan/Sign_Language_Gesture_Classifier)
* Save it in the project directory.

4. *Run inference*

* Open the notebook `SignLanguage.ipynb`
* Use the provided code cells to test predictions on your images.

---

# Usage Example

```python
from fastai.vision.all import load_learner, PILImage

# Load model
learn = load_learner('export2.pkl')

# Predict gesture
img = PILImage.create('sample_image.jpg')
pred, pred_idx, probs = learn.predict(img)

print(f"Predicted gesture: {pred}; Probability: {probs[pred_idx]:.04f}")
```

# contributing

Contributions, suggestions, and improvements are welcome!
Please fork the repo, create a branch, and submit a pull request.

---
