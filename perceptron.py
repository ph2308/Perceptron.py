import numpy as np
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
# Carregar o dataset de dígitos
digits = load_digits()
X, y = digits.data, digits.target

# Normalizar os dados
X = X / 16.0  # Os valores variam de 0 a 16

# Converter os rótulos para one-hot encoding
y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
# Função de ativação sigmoide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inicializar pesos e bias
input_size = X_train.shape[1]
hidden_size = 64  # Tamanho da camada oculta
output_size = y_train.shape[1]

np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
bias_output = np.zeros((1, output_size))

# Implementar o algoritmo de treinamento
def train(X, y, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, epochs=10000, lr=0.1):
    losses = []
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output = sigmoid(output_layer_input)
        
        # Calcular erro
        error = y - output
        
        # Backpropagation
        output_delta = error * sigmoid_derivative(output)
        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
        
        # Atualizar pesos e bias
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * lr
        bias_output += np.sum(output_delta, axis=0, keepdims=True) * lr
        weights_input_hidden += X.T.dot(hidden_delta) * lr
        bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * lr
        
        if epoch % 100 == 0:
            loss = np.mean(np.square(error))
            losses.append(loss)
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return losses

# Treinar o modelo
losses = train(X_train, y_train, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, epochs=5000, lr=0.01)
# Implementar o algoritmo de previsão
def predict(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)
    return np.argmax(output, axis=1)

# Prever e avaliar o modelo
y_pred_train = predict(X_train, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
y_pred_test = predict(X_test, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

# Converter rótulos one-hot de volta para valores inteiros
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calcular a precisão
train_accuracy = np.mean(y_pred_train == y_train_labels)
test_accuracy = np.mean(y_pred_test == y_test_labels)

print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
# Gráfico de perda
loss_fig = px.line(x=np.arange(len(losses)), y=losses, labels={'x': 'Epochs', 'y': 'Loss'}, title='Loss During Training')
loss_fig.show()
# Criar uma tabela de precisão
accuracy_df = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'Accuracy': [train_accuracy, test_accuracy]
})

# Exibir a tabela
print(accuracy_df)
# Calcular matriz de confusão
conf_matrix = confusion_matrix(y_test_labels, y_pred_test)

# Criar uma matriz de confusão plotly
conf_fig = ff.create_annotated_heatmap(conf_matrix, x=list(range(10)), y=list(range(10)), colorscale='Viridis')
conf_fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
conf_fig.show()
import numpy as np
import pandas as pd
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff # type: ignore

# Carregar o dataset de dígitos
digits = load_digits()
X, y = digits.data, digits.target

# Normalizar os dados
X = X / 16.0  # Os valores variam de 0 a 16

# Converter os rótulos para one-hot encoding
y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

# Função de ativação sigmoide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inicializar pesos e bias
input_size = X_train.shape[1]
hidden_size = 64  # Tamanho da camada oculta
output_size = y_train.shape[1]

np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
bias_output = np.zeros((1, output_size))

# Implementar o algoritmo de treinamento
def train(X, y, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, epochs=10000, lr=0.1):
    losses = []
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        output = sigmoid(output_layer_input)
        
        # Calcular erro
        error = y - output
        
        # Backpropagation
        output_delta = error * sigmoid_derivative(output)
        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
        
        # Atualizar pesos e bias
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * lr
        bias_output += np.sum(output_delta, axis=0, keepdims=True) * lr
        weights_input_hidden += X.T.dot(hidden_delta) * lr
        bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * lr
        
        if epoch % 100 == 0:
            loss = np.mean(np.square(error))
            losses.append(loss)
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return losses

# Treinar o modelo
losses = train(X_train, y_train, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, epochs=5000, lr=0.01)

# Implementar o algoritmo de previsão
def predict(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)
    return np.argmax(output, axis=1)

# Prever e avaliar o modelo
y_pred_train = predict(X_train, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
y_pred_test = predict(X_test, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

# Converter rótulos one-hot de volta para valores inteiros
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calcular a precisão
train_accuracy = np.mean(y_pred_train == y_train_labels)
test_accuracy = np.mean(y_pred_test == y_test_labels)

print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

# Visualizar resultados
# Gráfico de perda
loss_fig = px.line(x=np.arange(len(losses)), y=losses, labels={'x': 'Epochs', 'y': 'Loss'}, title='Loss During Training')
loss_fig.show()

# Criar uma tabela de precisão
accuracy_df = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'Accuracy': [train_accuracy, test_accuracy]
})

# Exibir a tabela
print(accuracy_df)

# Calcular matriz de confusão
conf_matrix = confusion_matrix(y_test_labels, y_pred_test)

# Criar uma matriz de confusão plotly
conf_fig = ff.create_annotated_heatmap(conf_matrix, x=list(range(10)), y=list(range(10)), colorscale='Viridis')
conf_fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
conf_fig.show()

#Resumo do código:
#Importei as bibliotecas necessárias.
#Carreguei o dataset de dígitos e normalizamos os dados.
#Implementei o perceptron com uma camada oculta e função de ativação sigmoide.
#Treinei o modelo com os dados de treinamento.
#Avaliei o modelo com os dados de teste e calculamos a precisão.











