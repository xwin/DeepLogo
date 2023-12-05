#! python

import os
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes

from PIL import Image
import numpy as np
import glob

from tensorflow.tools.graph_transforms import TransformGraph  # pylint: disable=g-import-not-at-top

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder
from object_detection.builders import post_processing_builder
from object_detection.core import box_list

# disable GPU use
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


MODEL_PATH='./frozen_inference_graph.pb'
CONFIG_PATH='./ssd_inception_v2_coco.config'
_DEFAULT_NUM_CHANNELS = 3
_DEFAULT_NUM_COORD_BOX = 4


def representative_data_gen():
    image_dir = "/home/alexp/work/nn/deeplogo/data/raw/"
    image_glob = "*_300.jpg"
    image_list = glob.glob(image_dir + image_glob)

    for imgfile in image_list:
        print("reading image {}".format(imgfile))
        img = Image.open(imgfile)
        input_image = np.array(img)
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
        input_image = np.subtract(np.multiply(input_image, 0.007843137718737125), 1)
        print(input_image.shape, input_image.dtype)
        yield [input_image]


def test_generator():
    gen = representative_data_gen()
    for img in gen:
        input_array = img
        print([list(s.shape) for s in input_array])
    return

def load_config(config_path):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf1.gfile.GFile(config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    return pipeline_config

def get_const_center_size_encoded_anchors(anchors):
    """Exports center-size encoded anchors as a numpy array.

    Args:
    anchors: a float32 tensor of shape [num_anchors, 4] containing the anchor
      boxes

    Returns:
    encoded_anchors: a float32 constant tensor of shape [num_anchors, 4]
    containing the anchor boxes.
    """
    anchor_boxlist = box_list.BoxList(anchors)
    y, x, h, w = anchor_boxlist.get_center_coordinates_and_sizes()
    num_anchors = y.get_shape().as_list()
    
    with tf.Session() as sess:
        y_out, x_out, h_out, w_out = sess.run([y, x, h, w])
        anchors_npy = np.transpose(np.stack((y_out, x_out, h_out, w_out)))

    return anchors_npy

# Run dummy prediction to populate anchors
def run_prediction(pipeline_config):
    detection_model = model_builder.build(
        pipeline_config.model, is_training=False)
    shape = [1, 300, 300, 3]
    image = tf.placeholder(
        tf.float32, shape=shape, name='normalized_input_image_tensor')
    predicted_tensors = detection_model.predict(image, true_image_shapes=None)
    return predicted_tensors

def output_tensor_names(pipeline_config, predicted_tensors):
    _, score_conversion_fn = post_processing_builder.build(
        pipeline_config.model.ssd.post_processing)
    class_predictions = score_conversion_fn(
        predicted_tensors['class_predictions_with_background'])
    output_tensors = list()
    output_tensors.append(predicted_tensors['box_encodings'].name[:-2])
    output_tensors.append('Postprocessor/'+ class_predictions.name[:-2])
    print(output_tensors)
    return output_tensors
    
def load_graph(model_filepath, anchors):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        graph = tf1.Graph()
        sess = tf1.InteractiveSession(graph = graph)

        with tf1.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf1.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n for n in graph_def.node if n.name in ('MultipleGridAnchorGenerator/Meshgrid_11/Shape')]
        for node in nodes:
            #print(node)
            pass

        anchor_node = graph_def.node.add()
        anchor_node.op = 'Const'
        anchor_node.name = 'anchors'
        anchor_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
        anchor_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(anchors)))        

        nodes = [n for n in graph_def.node if n.name in ('anchors')]
        for node in nodes:
            pass
            #print(node)

        # these valued are from ssd_inception_v2_coco.config
        max_detections = 10
        max_classes_per_detection = 1
        nms_score_threshold = [1e-8]
        nms_iou_threshold = [0.6]
        num_classes = 27
        scale_values = {}
        scale_values['y_scale'] = { 10.0 }
        scale_values['x_scale'] = { 10.0 }
        scale_values['h_scale'] = { 5.0 }
        scale_values['w_scale'] = { 5.0 }
        detections_per_class = 100
        use_regular_nms = False

        new_output = graph_def.node.add()
        new_output.op = 'TFLite_Detection_PostProcess'
        new_output.name = 'TFLite_Detection_PostProcess'
        new_output.attr['_output_quantized'].CopyFrom(
            attr_value_pb2.AttrValue(b=True))
        new_output.attr['_output_types'].list.type.extend([
            types_pb2.DT_FLOAT, types_pb2.DT_FLOAT, types_pb2.DT_FLOAT,
            types_pb2.DT_FLOAT
        ])
        new_output.attr['_support_output_type_float_in_quantized_op'].CopyFrom(
            attr_value_pb2.AttrValue(b=True))
        new_output.attr['max_detections'].CopyFrom(
            attr_value_pb2.AttrValue(i=max_detections))
        new_output.attr['max_classes_per_detection'].CopyFrom(
            attr_value_pb2.AttrValue(i=max_classes_per_detection))
        new_output.attr['nms_score_threshold'].CopyFrom(
            attr_value_pb2.AttrValue(f=nms_score_threshold.pop()))
        new_output.attr['nms_iou_threshold'].CopyFrom(
            attr_value_pb2.AttrValue(f=nms_iou_threshold.pop()))
        new_output.attr['num_classes'].CopyFrom(
            attr_value_pb2.AttrValue(i=num_classes))

        new_output.attr['y_scale'].CopyFrom(
            attr_value_pb2.AttrValue(f=scale_values['y_scale'].pop()))
        new_output.attr['x_scale'].CopyFrom(
            attr_value_pb2.AttrValue(f=scale_values['x_scale'].pop()))
        new_output.attr['h_scale'].CopyFrom(
            attr_value_pb2.AttrValue(f=scale_values['h_scale'].pop()))
        new_output.attr['w_scale'].CopyFrom(
            attr_value_pb2.AttrValue(f=scale_values['w_scale'].pop()))
        new_output.attr['detections_per_class'].CopyFrom(
            attr_value_pb2.AttrValue(i=detections_per_class))
        new_output.attr['use_regular_nms'].CopyFrom(
            attr_value_pb2.AttrValue(b=use_regular_nms))

        new_output.input.extend(
            ['Squeeze','Postprocessor/convert_scores','anchors'])

        input_names = []
        output_names = ['TFLite_Detection_PostProcess']
        transforms = ['strip_unused_nodes']

        transformed_graph_def = TransformGraph(graph_def, input_names,
                                               output_names, transforms)
        return transformed_graph_def


def main():

    pipeline_config = load_config(CONFIG_PATH)
    predicted_tensors = run_prediction(pipeline_config)
    anchors = get_const_center_size_encoded_anchors(predicted_tensors['anchors'])
    output_tensor_names(pipeline_config, predicted_tensors)
    graph_def = load_graph(MODEL_PATH, anchors)
    pb_filepath="updated_graph.pb"
    with tf1.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(graph_def.SerializeToString())

    return
    print("Saving tflite...")
    output_nodes = ['Squeeze','Postprocessor/convert_scores','anchors']
    output_nodes = ['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1',
                    'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3']
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=pb_filepath,
        input_arrays=['Preprocessor/sub'],
        output_arrays= output_nodes,
        input_shapes={'Preprocessor/sub' : [1, 300, 300, 3]}
    )
    
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if True :
        converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS ]
        converter.allow_custom_ops = True
    else:
        converter.target_spec.supported_ops = [ tf.compat.v1.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                                tf.lite.OpsSet.TFLITE_BUILTINS ]
        converter.representative_dataset = representative_data_gen
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open('deeplogo.tflite', 'wb') as f:
        f.write(tflite_model)
    
if __name__ == "__main__":
    main()
