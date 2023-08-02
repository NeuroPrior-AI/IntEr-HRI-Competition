import torch
from resnet import ResNet

if __name__ == "__main__":

    # Assuming that you are creating an instance of your model as follows:
    model = ResNet(num_classes=2)
    model.eval()

    # Create a dummy input that matches the input format of your model
    x = torch.randn(1, 64, 501)

    # Export the model to an ONNX file
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "./resnet.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    print(f'Done')
