# LSTM Text Generator (PyTorch)
I built a production-ready, character-level LSTM text generation system using PyTorch.  
This project was developed as part of a Generative AI engineering assessment.  
The solution is fully reproducible, modular, and demonstrates end-to-end ML workflow skills:
- Data preprocessing  
- Model design  
- Training with early stopping  
- Checkpointing  
- Deterministic text generation  
- Greedy, temperature, and beam search decoding  

---

## 🚀 Project Overview
This project trains a character-level LSTM on a large text corpus (e.g., Shakespeare).  
After training, the model can generate new text based on a seed sequence.

Main capabilities:
- Clean text preprocessing (lowercasing, filtering)
- Flexible sequence length
- Stable training loop with gradient clipping
- Checkpoint saving/restoring with vocab integrity
- Deterministic inference (`--mode greedy`)
- Creative generation (`--mode temp`)
- High-quality beam-search generation (`--mode beam`)

---

## 📁 Repository Structure
run---- paste in powershell
python lstm_text_generator.py `
  --data shakespeare.txt `
  --epochs 5 `
  --batch_size 64 `
  --seq_len 100 `
  --max_train_chars 300000 `
  --checkpoint lstm_ckpt.pt `
  --mode greedy `
  --gen_len 300
