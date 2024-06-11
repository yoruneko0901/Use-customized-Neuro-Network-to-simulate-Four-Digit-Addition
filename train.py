import numpy as np
import cupy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log.txt", mode='w', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

def generate_data(num_samples=10000):
    np.random.seed(42)
    X = np.random.randint(1000, 10000, (num_samples, 2))
    y = np.sum(X, axis=1).reshape(-1, 1)
    return X, y

class ShallowNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, init_method='xavier'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
        self.W1, self.b1, self.W2, self.b2 = self.initialize_weights(init_method)
        self.initialize_adam_parameters()

    def initialize_weights(self, method):
        if method == 'xavier':
            W1 = cp.random.randn(self.hidden_size, self.input_size) * cp.sqrt(1 / self.input_size)
            W2 = cp.random.randn(self.output_size, self.hidden_size) * cp.sqrt(1 / self.hidden_size)
        elif method == 'he':
            W1 = cp.random.randn(self.hidden_size, self.input_size) * cp.sqrt(2 / self.input_size)
            W2 = cp.random.randn(self.output_size, self.hidden_size) * cp.sqrt(2 / self.hidden_size)
        else:
            raise ValueError("init_method must be either 'xavier' or 'he'")
        
        b1 = cp.zeros((self.hidden_size, 1))
        b2 = cp.zeros((self.output_size, 1))
        return W1, b1, W2, b2
    
    def initialize_adam_parameters(self):
        self.mW1, self.vW1 = cp.zeros_like(self.W1), cp.zeros_like(self.W1)
        self.mb1, self.vb1 = cp.zeros_like(self.b1), cp.zeros_like(self.b1)
        self.mW2, self.vW2 = cp.zeros_like(self.W2), cp.zeros_like(self.W2)
        self.mb2, self.vb2 = cp.zeros_like(self.b2), cp.zeros_like(self.b2)
    
    def forward(self, x):
        self.z1 = cp.dot(self.W1, x) + self.b1
        self.a1 = cp.tanh(self.z1)
        self.z2 = cp.dot(self.W2, self.a1) + self.b2
        self.output = self.z2
        return self.output
    
    def predict(self, X):
        self.forward(X)
        return self.output
    
    def backward(self, x, y):
        m = x.shape[1]
        dz2 = self.output - y
        dW2 = (1/m) * cp.dot(dz2, self.a1.T)
        db2 = (1/m) * cp.sum(dz2, axis=1, keepdims=True)
        dz1 = cp.dot(self.W2.T, dz2) * (1 - cp.power(self.a1, 2))
        dW1 = (1/m) * cp.dot(dz1, x.T)
        db1 = (1/m) * cp.sum(dz1, axis=1, keepdims=True)
        self.update_weights(dW1, db1, dW2, db2)

    def update_weights(self, dW1, db1, dW2, db2):
        self.t += 1

        def adam_update(m, v, grad, beta1, beta2, epsilon, t):
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * cp.power(grad, 2)
            m_hat = m / (1 - cp.power(beta1, t))
            v_hat = v / (1 - cp.power(beta2, t))
            return m, v, m_hat / (cp.sqrt(v_hat) + epsilon)
        
        self.mW1, self.vW1, mW1_hat = adam_update(self.mW1, self.vW1, dW1, self.beta1, self.beta2, self.epsilon, self.t)
        self.mb1, self.vb1, mb1_hat = adam_update(self.mb1, self.vb1, db1, self.beta1, self.beta2, self.epsilon, self.t)
        self.mW2, self.vW2, mW2_hat = adam_update(self.mW2, self.vW2, dW2, self.beta1, self.beta2, self.epsilon, self.t)
        self.mb2, self.vb2, mb2_hat = adam_update(self.mb2, self.vb2, db2, self.beta1, self.beta2, self.epsilon, self.t)

        self.W1 -= self.learning_rate * mW1_hat
        self.b1 -= self.learning_rate * mb1_hat
        self.W2 -= self.learning_rate * mW2_hat
        self.b2 -= self.learning_rate * mb2_hat
        
    def train(self, X, y, X_val, y_val, epochs=1, batch_size=1024, patience=10):
        losses, val_losses = [], []
        best_val_loss, best_weights = float('inf'), None
        epochs_no_improve, stopped_epoch = 0, 0
        num_batches = (X.shape[1] + batch_size - 1) // batch_size
        total_steps = epochs * num_batches

        with tqdm(total=total_steps, desc="Training Progress") as bar:
            for epoch in range(epochs):
                for i in range(0, X.shape[1], batch_size):
                    x_batch = X[:, i:i + batch_size]
                    y_batch = y[:, i:i + batch_size]
                    self.forward(x_batch)
                    self.backward(x_batch, y_batch)
                    bar.update(1)

                y_pred_train = self.predict(X)
                loss = mean_squared_error(y.get(), y_pred_train.get())
                losses.append(loss)

                y_pred_val = self.predict(X_val)
                val_loss = mean_squared_error(y_val.get(), y_pred_val.get())
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss, best_weights = val_loss, (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    stopped_epoch = epoch
                    break

            if best_weights is not None:
                self.W1, self.b1, self.W2, self.b2 = best_weights
            
            return losses, val_losses, stopped_epoch
    def __str__(self):
        return f'\nW1={self.W1}\nb1={self.b1}\nW2={self.W2}\nb2={self.b2}'
    
    def save(self, file_path):
        np.savez(file_path, 
                    W1=self.W1.get(), 
                    b1=self.b1.get(), 
                    W2=self.W2.get(), 
                    b2=self.b2.get())
        logging.info(f"Model saved to {file_path}")

    def load(self, file_path):
        data = np.load(file_path)
        self.W1 = cp.array(data['W1'])
        self.b1 = cp.array(data['b1'])
        self.W2 = cp.array(data['W2'])
        self.b2 = cp.array(data['b2'])
        logging.info(f"Model loaded from {file_path}")
        
class Config:
    EPOCHS = 10
    BATCH_SIZE = pow(2, 4)
    NUM_SAMPLES = pow(2, 14)
    LEARNING_RATE = 0.001
    MIN_RANGE = 1
    MAX_RANGE = 20
    HIDDEN_SIZES = range(MIN_RANGE, MAX_RANGE + 1)
    ROUNDS = 30
    PATIENCE = int(EPOCHS*0.1)
    def __str__(self):
        return (
            f"Config:\n"
            f"  LEARNING_RATE={self.LEARNING_RATE}\n"
            f"  EPOCHS={self.EPOCHS}\n"
            f"  BATCH_SIZE={self.BATCH_SIZE}\n"
            f"  NUM_SAMPLES={self.NUM_SAMPLES}\n"
            f"  MIN_RANGE={self.MIN_RANGE}\n"
            f"  MAX_RANGE={self.MAX_RANGE}\n"
            f"  HIDDEN_SIZES={list(self.HIDDEN_SIZES)}\n"
            f"  ROUNDS={self.ROUNDS}\n"
            f"  PATIENCE={self.PATIENCE}\n"
            )
        
def check_and_save_data(csv_path, num_samples):
    if os.path.exists(csv_path):
        logging.info(f"Loading existing dataset from {csv_path}")
        data = pd.read_csv(csv_path)
        X = data[['x1', 'x2']].values
        y = data['y'].values.reshape(-1, 1)
    else:
        logging.info(f"No existing dataset found. Generating new data...")
        X, y = generate_data(num_samples)
        pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y.flatten()}).to_csv(csv_path, index=False)
        logging.info(f"Dataset saved to {csv_path}")
    return X, y

config = Config()
X, y = check_and_save_data('dataset.csv', config.NUM_SAMPLES)
train_size = int(0.8 * X.shape[0])
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler_X_0, scaler_X_1 = StandardScaler(), StandardScaler()
X_train_0 = scaler_X_0.fit_transform(X_train[:, 0].reshape(-1, 1))
X_train_1 = scaler_X_1.fit_transform(X_train[:, 1].reshape(-1, 1))
X_train = np.hstack((X_train_0, X_train_1))

X_test_0 = scaler_X_0.transform(X_test[:, 0].reshape(-1, 1))
X_test_1 = scaler_X_1.transform(X_test[:, 1].reshape(-1, 1))
X_test = np.hstack((X_test_0, X_test_1))

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

X_train_T = X_train.T
y_train_T = y_train.reshape(1, -1)

X_train = cp.array(X_train.T, dtype=cp.float32)
X_test = cp.array(X_test.T, dtype=cp.float32)
y_train = cp.array(y_train.T, dtype=cp.float32)
y_test = cp.array(y_test.T, dtype=cp.float32)

results = []
best_val_model, best_train_model = None, None
best_val_loss, best_train_loss = float('inf'), float('inf')
best_val_hidden_size, best_train_hidden_size = None, None
best_val_loss_history, best_train_loss_history = [], []

logging.info(config)
for hidden_size in config.HIDDEN_SIZES:
    logging.info(f'----Hidden_size {hidden_size}:')
    losses = []
    for _ in range(config.ROUNDS):
        logging.info(f'Round({_+1}/{config.ROUNDS})')
        nn = ShallowNeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1, learning_rate=config.LEARNING_RATE, init_method='xavier')
        loss_history, val_losses, epoch = nn.train(X_train, y_train, X_test, y_test, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, patience=config.PATIENCE)
        y_pred = nn.predict(X_test)
        loss = cp.sqrt(cp.mean(cp.square(y_pred - y_test)))
        losses.append(min(val_losses))
        if epoch == 0:
            logging.info(f'>>training completed\t\t| train loss={min(loss_history)}, val loss={min(val_losses)}')
        else:
            logging.info(f'>>early stopped at epoch {epoch+1}\t| train loss={min(loss_history)}, val loss={min(val_losses)}')

        if min(val_losses) < best_val_loss:
            best_val_loss = min(val_losses)
            best_val_model = nn
            best_val_hidden_size = hidden_size
            best_val_loss_history = val_losses
            best_val_model.save('best_val_model.npz')  # 保存最佳驗證模型
        if min(loss_history) < best_train_loss:
            best_train_loss = min(loss_history)
            best_train_model = nn
            best_train_hidden_size = hidden_size
            best_train_loss_history = loss_history
            best_train_model.save('best_train_model.npz')  # 保存最佳訓練模型

    results.append(losses)
    logging.info(f'#目前最低損失: train={best_train_loss}, val={best_val_loss}')
    logging.info(f'#目前最佳隱藏層神經元數量: train={best_train_hidden_size}, val={best_val_hidden_size}')
    logging.info(f'#目前最佳參數:\n-train:{best_train_model}\n-val:{best_val_model}\n')
logging.info(f'----最佳隱藏層神經元數量: train={best_train_hidden_size}, val={best_val_hidden_size}')

plt.figure(figsize=(12, 6))
plt.boxplot(results)
plt.xlabel('Hidden Layer Size')
plt.ylabel('Loss')
plt.title('Model Performance for Different Hidden Layer Sizes')
plt.savefig('Model Performance for Different Hidden Layer Sizes.png')

best_val_model.forward(X_test)
y_pred = best_val_model.output
y_test_inv = scaler_y.inverse_transform(y_test.get().reshape(-1, 1)).flatten()
y_pred_inv = scaler_y.inverse_transform(y_pred.get().reshape(-1, 1)).flatten()

plt.figure(figsize=(18, 6))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Sum')
plt.title(f'Best Model Predictions vs True Values (Hidden Size: {best_val_hidden_size})')
plt.legend()
plt.savefig('Best Model Predictions vs True Values (val).png')

best_train_model.forward(X_test)
y_pred = best_train_model.output
y_test_inv = scaler_y.inverse_transform(y_test.get().reshape(-1, 1)).flatten()
y_pred_inv = scaler_y.inverse_transform(y_pred.get().reshape(-1, 1)).flatten()

plt.figure(figsize=(18, 6))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Sum')
plt.title(f'Best Model Predictions vs True Values (Hidden Size: {best_train_hidden_size})')
plt.legend()
plt.savefig('Best Model Predictions vs True Values (train).png')

plt.figure(figsize=(6, 6))
plt.scatter(y_test_inv, y_pred_inv, alpha=0.6, label='Predicted vs Actual')
plt.plot([min(y_test_inv), max(y_test_inv)], [min(y_test_inv), max(y_test_inv)], 'r--', label='Ideal Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values with Best Hidden Layer Size')
plt.legend()
plt.savefig('Actual vs Predicted Values with Best Hidden Layer Size.png')

plt.figure(figsize=(12, 6))
plt.plot(best_val_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Learning Curve (Hidden Size: {best_val_hidden_size})')
plt.savefig('Learning Curve (val).png')

plt.figure(figsize=(12, 6))
plt.plot(best_train_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Learning Curve (Hidden Size: {best_train_hidden_size})')
plt.savefig('Learning Curve (train).png')

errors = y_pred_inv - y_test_inv
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=50)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title(f'Error Histogram (Hidden Size: {best_val_hidden_size})')
plt.savefig('Error Histogram (val).png')

mean_losses = [np.mean(losses) for losses in results]
plt.figure(figsize=(12, 6))
plt.plot(config.HIDDEN_SIZES, mean_losses, marker='o')
plt.xlabel('Number of Hidden Layer Neurons')
plt.ylabel('Mean Squared Error')
plt.title('Hidden Layer Size vs. MSE')
plt.savefig('Hidden Layer Size vs MSE')