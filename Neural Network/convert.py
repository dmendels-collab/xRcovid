import coremltools as ct
from tensorflow.keras.models import load_model
import argparse
from coremltools.models.neural_network import quantization_utils


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

print("[INFO] loading model...")
model = load_model(args["model"])

print("[INFO] converting model...")
mlmodel = ct.convert(model)
spec = mlmodel.get_spec()
ct.utils.rename_feature(spec, 'Identity', 'confidence')
ct.utils.rename_feature(spec, 'conv2d_input', 'image')
mlmodel = ct.models.MLModel(spec)

mlmodel.author = 'xRapid Group'
mlmodel.license = 'Private Use'
mlmodel.short_description = 'Classifies RDTs.'
mlmodel.version = '1.0.0'

mlmodel.input_description['image'] = 'Image. Grayscale. Normalised. Shape: (1, 256, 256, 1). Type: float32.'
mlmodel.output_description['confidence'] = '0=negative. 1=positive. Shape: (1). Type: float32.'

print('[INFO] quantizing model...')
mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8)

print('[INFO] saving model...')
mlmodel.save('covidNet.mlmodel')
