# Imports here
import torch
from torch import nn,optim
from torchvision import models
from  PIL import Image
import numpy as np
import argparse
import json



# Loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']    
    model = eval("models.{}(pretrained=True)".format(arch))
               
    model.class_to_idx = checkpoint['classes_indices_map']
    if checkpoint.has_key('classifier'):
        model.classifier = checkpoint['classifier']
    elif checkpoint.has_key('fc'):
        model.fc = checkpoint['fc']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    
# Freeze the feature parameters
    for params in model.parameters():
        params.requires_grad = False
       
    return model,optimizer,epoch


#Process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Resize the images using thumbnail to keep the aspect ratios
    resize = (256,256)
    img = Image.open(image)
    img.thumbnail(resize)
    #print("After thumbnail: ",img.size)
    
    #Crop out the center 224x224 portion of the image    
    width, height = img.size  # Get dimensions
    new_width = 224
    new_height = 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    #print("Cropped Image size: ",img.size)
    
    #Color channels of images
    np_image = np.array(img)/255
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.485, 0.456, 0.406])
    #print(type(np_image))
    
    #Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))
    #print(np_image.shape)
    
    return np_image



#Predict the image
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.cpu()
    #Convert Numpy array to tensor
    img_tensor = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor)
    #Correct dimensions
    img_dim = img_tensor.unsqueeze_(0)
        
    #Calculate probs and classes in evaluation mode and turn off gradients    
    model.eval()
    with torch.no_grad():                
        log_ps = model.forward(img_dim)
        ps = torch.exp(log_ps)
        
        top_ps,indices = ps.topk(topk)
        top_ps = top_ps.tolist()[0]
        indices = indices.tolist()[0]
        
        mapping = {name:key for key,name in model.class_to_idx.items()}
        classes = [mapping[index] for index in indices]
        
    return top_ps,classes


#Main function
def main():
    # Define Command Line arguments for the script
    parser = argparse.ArgumentParser (description = "Predicts the class for an input image using a trained network.")

    parser.add_argument ('imagefile', help = 'Provide path of the image file.', type = str, action="store", default='./flowers/test/100/image_07896.jpg')
    parser.add_argument ('checkpoint', help = 'Provide path of the checkpoint. Default is current directory', type = str, action="store", default="./model_checkpoint.pth")
    parser.add_argument ('--top_k', help = 'Optional: Return top K most likely classes. Default value is 5.', type = int, dest="top_k", action="store", default=5)
    parser.add_argument ('--category_names', help = 'Optional: Provide path of file that has mapping of categories to real names. Default is ./cat_to_name.json', type =str, dest="category_names", action="store", default='./cat_to_name.json')
    parser.add_argument ('--GPU', help = "Optional: Option to use GPU. Default is GPU", type = str, dest="GPU", action="store", default="GPU")

    #setting values data loading
    args = parser.parse_args ()
    
    #assign values
    image_path = args.imagefile    
    checkpoint_path = args.checkpoint
    topk = args.top_k
    category_names_file = args.category_names
    
    
    if args.GPU == 'GPU':
        device = 'cuda'
    else:
        device = 'cpu'
        
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)

    print("Loading the pretrained model from the checkpoint: {} ...".format(checkpoint_path))
    model,optimizer,epoch = load_checkpoint(checkpoint_path)
    print("Successfully loaded the model.")
    print("Optimizer: ",optimizer)
    print("Model: ",model)
    
    model.to(device)
    
    print("Predicting the name of the flower in the image and its probability...")
    probs, classes = predict(image_path, model, topk)
    flower_names = [cat_to_name[i] for i in classes]
    
    for name, value in zip(flower_names, probs):
        print ("Flower name: {} and its probability: {}".format(name, value))
            
    model.cpu()    
        
if __name__ == '__main__':
    main()