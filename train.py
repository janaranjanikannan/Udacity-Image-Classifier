# Imports here
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import argparse


#Load the data
def get_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validate_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validate_datasets = datasets.ImageFolder(valid_dir, transform=validate_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validateloader = torch.utils.data.DataLoader(validate_datasets, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

    return train_datasets,trainloader, validateloader, testloader


#Build the network
def build_network(input_size, output_size, hidden_units, drop_p,arch):        

    model = eval("models.{}(pretrained=True)".format(arch))
    
    for params in model.parameters():
        params.requires_grad= False
            
    fc_classifier = nn.Sequential(nn.Linear(input_size,hidden_units), 
                                     nn.ReLU(),
                                     nn.Dropout(drop_p),
                                     nn.Linear(hidden_units,output_size),
                                     nn.LogSoftmax(dim=1))
    if arch.startwith('resnet'):
        model.fc = fc_classifier 
    else:
        model.classifier = fc_classifier
    
    criterion = nn.NLLLoss()
        
    return model,criterion


#Get optimizer
def get_optimizer(model,learning_rate):
    return optim.Adam(model.classifier.parameters(),lr=learning_rate)


#Train the network
def train_network(model, trainloader, validateloader, criterion, optimizer, device, epochs, print_every):
    steps = 0
    train_loss = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                validate_loss = 0
                validate_accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validateloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        loss = criterion(log_ps, labels)

                        validate_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validate_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {train_loss/print_every:.3f}.. "
                      f"Validation loss: {validate_loss/len(validateloader):.3f}.. "
                      f"Validation accuracy: {validate_accuracy/len(validateloader):.3f}")
                train_loss = 0
                model.train()
    print("Training completed successfully!")
    

#Save the model
def save_model(model, arch, optimizer, train_datasets, save_dir, input_size, output_size, hidden_units, epochs):
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'classes_indices_map': model.class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),              
                  'epoch': epochs}
    
    if arch.startswith('resnet):
        checkpoint['fc'] = model.fc
    else:
        checkpoint['classifier'] = model.classifier

    model_checkpoint=save_dir+'/model_checkpoint.pth'
    torch.save(checkpoint, model_checkpoint)
    print("Model is saved. The location of {} is {}".format('model_checkpoint.pth',save_dir))
    

#Main function
def main():
    # Define Command Line arguments for the script
    parser = argparse.ArgumentParser (description = "Train a new network on a dataset and save the model as a checkpoint")

    parser.add_argument ('data_dir', help = 'Provide data directory.', type = str, action="store", default='./flowers')
    parser.add_argument ('--save_dir', help = 'Optional: Provide saving directory. Default is current directory', type = str, dest="save_dir", action="store", default=".")
    parser.add_argument ('--arch', help = 'Optional: Currently it supports only densenet121, densenet169, densenet161, densenet201 and alexnet pretrained model. Default pretrained model is densenet121', type = str, dest="arch", action="store", default="densenet121")
    parser.add_argument ('--learning_rate', help = 'Optional: Learning rate. Default value is 0.003', type = float,dest="learning_rate", action="store", default=0.003)
    parser.add_argument ('--input_units', help = 'Optional: Input units in Classifier. Default value is 1024', type = int, dest="input_units", action="store", default=1024)
    parser.add_argument ('--hidden_units', help = 'Optional: Hidden units in Classifier. Default value is 512', type = int, dest="hidden_units", action="store", default=512)
    parser.add_argument ('--output_units', help = 'Optional: Hidden units in Classifier. Default value is 102', type = int, dest="output_units", action="store", default=102)
    parser.add_argument ('--epochs', help = 'Optional: Number of epochs. Default value is 5', type = int, dest="epochs", action="store",  default=6)
    parser.add_argument ('--GPU', help = "Optional: Option to use GPU. Default is GPU", type = str, dest="GPU", action="store", default="GPU")

    #setting values data loading
    args = parser.parse_args ()
    
    #assign values
    data_dir = args.data_dir    
    save_dir = args.save_dir    
    arch = args.arch    
    learning_rate = args.learning_rate
    input_size = args.input_units
    hidden_units = args.hidden_units
    output_size = args.output_units
    epochs = args.epochs
    
    if arch == 'alexnet':
        if input_size != 9216:
            print("{} is incorrect number of input units for the pretrained model: {}. Hence setting it to 9216.".format(input_size,arch))
    
    if arch == 'densenet121':
        if input_size != 1024:
            print("{} is incorrect number of input units for the pretrained model: {}. Hence setting it to 1024.".format(input_size,arch))
    
    if arch == 'densenet169':
        if input_size != 1664:
            print("{} is incorrect number of input units for the pretrained model: {}. Hence setting it to 1664.".format(input_size,arch))
    
    if arch == 'densenet161':
        if input_size != 2208:
            print("{} is incorrect number of input units for the pretrained model: {}. Hence setting it to 2208.".format(input_size,arch))
    
    if arch == 'densenet201':
        if input_size != 1920:
            print("{} is incorrect number of input units for the pretrained model: {}. Hence setting it to 1920.".format(input_size,arch))

    if args.GPU == 'GPU':
        device = 'cuda'
    else:
        device = 'cpu'

    drop = 0.5
    print_every = 40
        
    print("Loading the data...")
    train_datasets,trainloader, validateloader, testloader = get_data(data_dir)
    
    print("Building the network...")
    model,criterion = build_network(input_size, output_size, hidden_units, drop, arch)
    print("Model:", model)
    model.to(device)
    
    print("Getting the optimizer...")
    optimizer = get_optimizer(model,learning_rate)
    print("Optimizer: ", optimizer)
    
    print("Training the model...")
    train_network(model, trainloader, validateloader, criterion, optimizer, device, epochs=5, print_every=40)
    
    print("Saving the model..")
    save_model(model, arch, optimizer, train_datasets, save_dir, input_size, output_size, hidden_units,epochs)    
    model.cpu()
    
if __name__ == '__main__':
    main()