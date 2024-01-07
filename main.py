# Mini-project for D7041E
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

torch.manual_seed(0)
batch_size_train = 64
batch_size_test = 1000

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)

class Cross_Entropy_Model(nn.Module):
    def __init__(self, num_hidden_layers, hidden_layer_sizes):
        super(Cross_Entropy_Model, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = nn.ModuleList()

        # Input layer
        self.hidden_layers.append(nn.Linear(28 * 28, hidden_layer_sizes[0]))

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))

        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        
        x = self.output_layer(x)
        return x

class NLL_model(nn.Module):
    def __init__(self, num_hidden_layers, hidden_layer_sizes):
        super(NLL_model, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layers = nn.ModuleList()

        # Input layer
        self.hidden_layers.append(nn.Linear(28 * 28, hidden_layer_sizes[0]))

        # Hidden layers
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))

        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            # x = torch.relu(layer(x))
            x = nn.functional.relu(layer(x))
        
        x = self.output_layer(x)
        return x

def train_cross_entropy_model(criterion, epochs, num_layers_list, layer_sizes_list):
    models = []
    
    for num_hidden_layers, hidden_layer_sizes in zip(num_layers_list, layer_sizes_list):
        print(f'Number of hidden layers: {num_hidden_layers-1}, Hidden layer sizes: {hidden_layer_sizes}')
        for hidden_layer_size in hidden_layer_sizes:
            print(f'Current layer size: {hidden_layer_size}')
            model = Cross_Entropy_Model(num_hidden_layers, hidden_layer_size)
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(epochs):
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')
            
            models.append((model, num_hidden_layers, hidden_layer_size))
    return models

def train_NLL_model(criterion, epochs, num_layers_list, layer_sizes_list):
    models = []
    
    for num_hidden_layers, hidden_layer_sizes in zip(num_layers_list, layer_sizes_list):
        print(f'Number of hidden layers: {num_hidden_layers-1}, Hidden layer sizes: {hidden_layer_sizes}')
        for hidden_layer_size in hidden_layer_sizes:
            print(f'Hidden layer sizes: {hidden_layer_sizes}')
            model = NLL_model(num_hidden_layers, hidden_layer_size)
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            for epoch in range(epochs):
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    outputs = torch.nn.functional.log_softmax(outputs, dim=1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')
            
            models.append((model, num_hidden_layers, hidden_layer_size))
    return models

epochs = 4
num_layers_list = [1, 2, 3]
layer_sizes_list = [[(256,), (512,), (1024,)], [(256, 256), (512, 512), (1024, 1024)], [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]]

criterion = nn.CrossEntropyLoss()
models_L1Loss = train_cross_entropy_model(criterion, epochs, [1, 2, 3], layer_sizes_list)


criterion = nn.NLLLoss()
model = train_NLL_model(criterion, epochs, [1, 2, 3], layer_sizes_list)