import torchvision
import torch
import time


# Here we load the Vgg19 model,and in the forward pass only return the feature maps for the layers specified via the  content and style layer arguments passed when intialzing the class object.
class vgg19(torch.nn.Module):
    def __init__(self,style_layers,content_layer):
        super().__init__()

        self.style_layers=style_layers
        self.content_layer=content_layer
        self.layers=torch.nn.ModuleDict()
        model=torchvision.models.vgg19(weights="DEFAULT")    
        layer=1
        sub_layer=1
        relu_class_name=type(model.features[1])
        maxpool_class_name=type(model.features[4])
        features=model.features
        for x in features:
            if(isinstance(x,maxpool_class_name)):
                layer+=1
                sub_layer=1
                
            dict_key="conv"+str(layer)+"_"+str(sub_layer)
            if dict_key not in self.layers:
                self.layers[dict_key]=torch.nn.Sequential()
            self.layers[dict_key].add_module(str(x),x)
            
            if(isinstance(x,torch.nn.modules.activation.ReLU)):
                sub_layer+=1

        self.transforms=torchvision.transforms.Compose([torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms()])
    
    def forward(self,img):
        output={"style_feature_map":{}, "content_feature_map":{}}
        res=img
        for layer in self.layers:
            res=self.layers[layer](res)
            
            if layer in self.content_layer:
                output["content_feature_map"][layer]=res
            
            if layer in self.style_layers:
                output["style_feature_map"][layer]=res

            
        return  output
                                     
            
    