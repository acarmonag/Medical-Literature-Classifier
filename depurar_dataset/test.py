import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Cargar el dataset
df = pd.read_csv('medical_lit.csv', sep=';')

# Analizar la distribución de las etiquetas
def analyze_label_distribution(df):
    # Extraer todas las etiquetas individuales
    all_labels = []
    for group in df['group']:
        if pd.notna(group):
            labels = [label.strip() for label in group.split('|')]
            all_labels.extend(labels)
    
    # Contar frecuencias
    label_counts = Counter(all_labels)
    
    print("=== ANÁLISIS DE BALANCE DEL DATASET ===\n")
    print(f"Total de documentos: {len(df)}")
    print(f"Número de etiquetas únicas: {len(label_counts)}")
    print(f"Total de asignaciones de etiquetas: {sum(label_counts.values())}")
    
    # Mostrar distribución de etiquetas
    print("\n--- DISTRIBUCIÓN POR ETIQUETA ---")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(df)) * 100
        print(f"{label:<15}: {count:>3} documentos ({percentage:>5.1f}%)")
    
    # Analizar desbalance
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    ratio = max_count / min_count
    
    print(f"\n--- MÉTRICAS DE BALANCE ---")
    print(f"Etiqueta más frecuente: {max(label_counts, key=label_counts.get)} ({max_count} docs)")
    print(f"Etiqueta menos frecuente: {min(label_counts, key=label_counts.get)} ({min_count} docs)")
    print(f"Ratio desbalance: {ratio:.2f}:1")
    
    # Clasificar nivel de desbalance
    if ratio <= 2:
        balance_level = "BIEN BALANCEADO"
    elif ratio <= 5:
        balance_level = "MODERADAMENTE DESBALANCEADO"
    elif ratio <= 10:
        balance_level = "DESBALANCEADO"
    else:
        balance_level = "SEVERAMENTE DESBALANCEADO"
    
    print(f"Nivel de balance: {balance_level}")
    
    return label_counts

# Analizar distribución de combinaciones multi-etiqueta
def analyze_multilabel_patterns(df):
    print("\n--- ANÁLISIS MULTI-ETIQUETA ---")
    
    label_combinations = []
    single_labels = 0
    multi_labels = 0
    
    for group in df['group']:
        if pd.notna(group):
            labels = [label.strip() for label in group.split('|')]
            if len(labels) == 1:
                single_labels += 1
            else:
                multi_labels += 1
            label_combinations.append(tuple(sorted(labels)))
    
    combination_counts = Counter(label_combinations)
    
    print(f"Documentos con una sola etiqueta: {single_labels} ({single_labels/len(df)*100:.1f}%)")
    print(f"Documentos con múltiples etiquetas: {multi_labels} ({multi_labels/len(df)*100:.1f}%)")
    
    print(f"\nCombinaciones más comunes:")
    for combo, count in sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        combo_str = " + ".join(combo)
        print(f"  {combo_str:<30}: {count:>3} documentos")

# Ejecutar análisis
label_counts = analyze_label_distribution(df)
analyze_multilabel_patterns(df)

# Crear visualización
plt.figure(figsize=(12, 8))

# Gráfico de barras para distribución de etiquetas
plt.subplot(2, 2, 1)
labels, counts = zip(*sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(labels)])
plt.title('Distribución de Etiquetas')
plt.xlabel('Etiquetas')
plt.ylabel('Número de Documentos')
plt.xticks(rotation=45)

# Añadir valores en las barras
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom')

# Gráfico de pastel
plt.subplot(2, 2, 2)
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Proporción de Etiquetas')

# Histograma de longitud de textos
plt.subplot(2, 2, 3)
text_lengths = df['abstract'].fillna('').str.len()
plt.hist(text_lengths, bins=30, alpha=0.7, color='skyblue')
plt.title('Distribución de Longitud de Abstracts')
plt.xlabel('Caracteres')
plt.ylabel('Frecuencia')

# Análisis de co-ocurrencia
plt.subplot(2, 2, 4)
# Matriz de co-ocurrencia simplificada
unique_labels = list(label_counts.keys())
n_labels = len(unique_labels)
cooccurrence_matrix = np.zeros((n_labels, n_labels))

for group in df['group']:
    if pd.notna(group):
        labels = [label.strip() for label in group.split('|')]
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if label1 in labels and label2 in labels and i != j:
                    cooccurrence_matrix[i][j] += 1

sns.heatmap(cooccurrence_matrix, xticklabels=unique_labels, yticklabels=unique_labels, 
            annot=True, fmt='.0f', cmap='Blues')
plt.title('Co-ocurrencia de Etiquetas')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# Recomendaciones para balancear el dataset
print("\n=== RECOMENDACIONES PARA BALANCEAR ===")
max_count = max(label_counts.values())
min_count = min(label_counts.values())

for label, count in sorted(label_counts.items(), key=lambda x: x[1]):
    if count < max_count * 0.5:  # Si tiene menos del 50% de la clase mayoritaria
        needed = max_count - count
        print(f"• {label}: necesita ~{needed} documentos adicionales")

print(f"\nTécnicas recomendadas:")
print(f"• Data Augmentation para clases minoritarias")
print(f"• Oversampling (SMOTE) para balancear")
print(f"• Weighted loss functions durante entrenamiento")
print(f"• Stratified sampling para train/val/test splits")