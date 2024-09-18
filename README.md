# QA_Training_Model
This project uses a BART model and T5 model for answering finance-related questions. The model is trained using a dataset of financial text and summaries.  
## Files  

- `train_t5_model.py`: Script to train the T5 model.
- `gradio_t5_interface.py`: Gradio interface for asking questions and receiving answers.
- `requirements.txt`: List of required Python libraries.
- `train_model.py`: Script to train the BART model.
- `gradio_interface.py`: Gradio interface for asking questions and receiving answers.  

## Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```
   OR
   ```bash
   python train_t5_model.py
   ```

3. Launch the Gradio interface:
   ```bash
   python gradio_interface.py
   ```
   OR
   ```bash
   python gradio_t5_interface.py
   ```
## Dataset
The dataset used contains two columns: text (questions) and summary (answers).
   
