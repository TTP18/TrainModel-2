import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pyvi import ViTokenizer 

# Đọc dữ liệu từ tệp
def load_test(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        text = file.read()
    return text

file_path = 'C:/Users/ADMIN/Desktop/NLP_TL/demo.txt' 
# Văn bản đầu vào 
text = load_test(file_path)

# Load stopwords tiếng Việt
stop_words_vietnamese = {
    "và", "là", "của", "có", "trong", "khi", "được", "cho", "này", "đó",
    "một", "những", "rằng", "với", "về", "tại", "thì", "như", "nếu", "ra",
    "tôi", "anh", "chị", "ông", "bà", "chúng", "họ", "để", "cũng", "nhiều",
    "hơn", "nào", "điều", "đã", "càng", "làm", "phải", "thế", "nơi"
}
stopwords = stop_words_vietnamese  # Bổ sung danh sách stopwords phù hợp

# Tiền xử lý văn bản cơ bản
def preprocess_text(text):
    text = text.lower() # Chuyển đổi chữ thường
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Loại bỏ URL
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu và ký tự đặc biệt
    text = re.sub(r'\s+',' ',text).strip() # Loại bỏ khoảng trắng thừa
    words = ViTokenizer.tokenize(text)  # Tách từ bằng pyvi
    words = words.split() # Chuyển kết quả thành danh sách từ
    if stopwords:
        words = [word for word in words if word not in stopwords] # Loại bỏ stopwords
    return words

# Tiền xử lý đoạn văn
processed_sentences = [preprocess_text(sentence) for sentence in text.split(".") if sentence.strip() != ""]

# Đếm số từ trong text
def count_words(text):
    # Loại bỏ dấu câu và ký tự đặc biệt
    text = re.sub(r'[^\w\s]', '', text)
    # Tách từ
    words = ViTokenizer.tokenize(text).split()
    # Đếm số từ
    return len(words)

word_count = count_words(text)
print(f"Tổng số từ trong đoạn văn: {word_count}")

# Tạo từ điển từ vựng và ánh xạ từ-vị trí
def build_vocab(sentences):
    word_counts = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            word_counts[word] += 1
    vocab = {word: i for i, word in enumerate(word_counts.keys())}
    reverse_vocab = {i: word for word, i in vocab.items()}
    return vocab, reverse_vocab

# Xây dựng từ điển
vocab, reverse_vocab = build_vocab(processed_sentences)

# Tạo dữ liệu context-target cho Skip-gram
def generate_training_data(sentences, vocab, window_size=2):
    data = []
    for sentence in sentences:
        indices = [vocab[word] for word in sentence if word in vocab]
        for center_pos in range(len(indices)):
            for w in range(-window_size, window_size + 1):
                context_pos = center_pos + w
                if context_pos < 0 or context_pos >= len(indices) or center_pos == context_pos:
                    continue
                data.append((indices[center_pos], indices[context_pos]))
    return np.array(data)

# Xây dựng dữ liệu huấn luyện 
training_data = generate_training_data(processed_sentences, vocab)

# Skip-gram model
class SkipGramModel:
    def __init__(self, vocab_size, embed_size, learning_rate=0.03):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        # Ma trận nhúng [vocab_size x embed_size] và ma trận trọng số đầu ra [embed_size x vocab_size]
        self.W1 = np.random.uniform(-0.8, 0.8, (vocab_size, embed_size))  # Input embedding
        self.W2 = np.random.uniform(-0.8, 0.8, (embed_size, vocab_size))  # Output embedding

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def forward(self, center_word_idx):
        h = self.W1[center_word_idx]  # Vector nhúng của từ trung tâm
        u = np.dot(self.W2.T, h)  # Dự đoán phân phối
        y_pred = self.softmax(u)
        return y_pred, h

    def backward(self, error, h, center_word_idx):
        # Gradient descent update
        dl_dw2 = np.outer(h, error)
        dl_dw1 = np.dot(self.W2, error)
        self.W1[center_word_idx] -= self.learning_rate * dl_dw1
        self.W2 -= self.learning_rate * dl_dw2
            
    def train(self, training_data, epochs=10):
        loss_history = []
        for epoch in range(epochs):
            loss = 0
            for center_word, context_word in training_data:
                y_pred, h = self.forward(center_word)
                # Tạo nhãn one-hot cho từ ngữ cảnh
                y_true = np.zeros(self.vocab_size)
                y_true[context_word] = 1
                # Tính toán lỗi
                error = y_pred - y_true
                # Cập nhật trọng số                
                self.backward(error, h, center_word)
                # Hàm mất mát cross-entropy
                loss += -np.log(y_pred[context_word])
            loss_history.append(loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        return loss_history

# Vẽ biểu đồ mất mátS
def plot_loss(loss_history):
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.show()


# Đánh giá vector nhúng
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def evaluate_embeddings(model, word_pairs, vocab):
    results = {}
    for word1, word2 in word_pairs:
        if word1 in vocab and word2 in vocab:
            idx1 = vocab[word1]
            idx2 = vocab[word2]
            vec1 = model.W1[idx1]
            vec2 = model.W1[idx2]
            similarity = cosine_similarity(vec1, vec2)
            results[(word1, word2)] = similarity
    return results

# Tính ma trận cosine similarity
def compute_similarity_matrix(embeddings):
    vocab_size = embeddings.shape[0]
    similarity_matrix = np.zeros((vocab_size, vocab_size))
    for i in range(vocab_size):
        for j in range(vocab_size):
            similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])
    return similarity_matrix



# Vẽ ma trận tương quan
def plot_similarity_matrix(similarity_matrix, reverse_vocab):
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix_10, annot=True, cmap="coolwarm", xticklabels=selected_words, yticklabels=selected_words)
    plt.title("Cosine Similarity Heatmap for 10 Words")
    plt.xlabel("Words")
    plt.ylabel("Words")
    plt.show()

print("Vocabulary size:",len(vocab))
print("Example Vocabulary:",list(vocab.items())[:10])
print("Example Tranning_Data:",training_data[:10])

selected_words = list(vocab.keys())[:10]
indices = [vocab[word] for word in selected_words]



# Huấn luyện mô hình
embed_size = 50
model = SkipGramModel(vocab_size=len(vocab), embed_size=embed_size)
loss_history = model.train(training_data, epochs=20)

# Tính ma trận và vẽ
selected_embeddings = model.W1[indices]  # Lấy ma trận nhúng đầu vào
similarity_matrix_10 = compute_similarity_matrix(selected_embeddings)
plot_similarity_matrix(similarity_matrix_10, reverse_vocab)


# Vẽ biểu đồ mất mát
plot_loss(loss_history)


# Đánh giá mô hình
word_pairs = [("mạng","xã_hội"),("nổi_tiếng","người"),("điện_ảnh","âm_nhạc"),("tự_tin","tự_ti"),("kiểm_soát","kiểm_duyệt")]
embedding_evaluation = evaluate_embeddings(model, word_pairs, vocab)
for pair, sim in embedding_evaluation.items():
    if pair[0] not in vocab or pair[1] not in vocab:
        print(f"Error: One of the words {pair[0]} or {pair[1]} is not in vocab.")
    else:
        print(f"Similarity between {pair[0]} and {pair[1]}: {sim:.4f}")















