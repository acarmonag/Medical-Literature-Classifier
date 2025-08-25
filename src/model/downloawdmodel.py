from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
save_dir = "local_pubmedbert"

# Descarga y guarda el modelo y el tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)