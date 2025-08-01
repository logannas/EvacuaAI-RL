# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: src/grpc_server/evacuai_rl.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'src/grpc_server/evacuai_rl.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n src/grpc_server/evacuai_rl.proto\x12\x02rl\"\x90\x01\n\x0cTrainRequest\x12\x12\n\nproject_id\x18\x01 \x01(\t\x12&\n\x19transfer_learning_version\x18\x02 \x01(\tH\x00\x88\x01\x01\x12&\n\x0fhyperparameters\x18\x03 \x01(\x0b\x32\r.rl.HypParamsB\x1c\n\x1a_transfer_learning_version\"!\n\rTrainResponse\x12\x10\n\x08model_id\x18\x01 \x01(\t\"}\n\x10InferenceRequest\x12\x12\n\nproject_id\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x16\n\x0eprevious_state\x18\x03 \x01(\x05\x12\x12\n\nfire_nodes\x18\x04 \x03(\x05\x12\x18\n\x10\x61gents_positions\x18\x05 \x03(\x05\"w\n\x14GetRewardPathRequest\x12\x12\n\nproject_id\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x0c\n\x04path\x18\x03 \x03(\x05\x12\x12\n\nfire_nodes\x18\x04 \x03(\x05\x12\x18\n\x10\x61gents_positions\x18\x05 \x03(\x05\":\n\x04Path\x12\x11\n\tinit_node\x18\x01 \x01(\x05\x12\x11\n\tlast_node\x18\x02 \x01(\x05\x12\x0c\n\x04path\x18\x03 \x03(\x05\"2\n\x11InferenceResponse\x12\x1d\n\x0bpredictions\x18\x01 \x03(\x0b\x32\x08.rl.Path\"\'\n\x15GetRewardPathResponse\x12\x0e\n\x06reward\x18\x01 \x01(\x02\"\xcf\x04\n\tHypParams\x12\x11\n\x04\x62\x65ta\x18\x01 \x01(\x02H\x00\x88\x01\x01\x12\x0f\n\x02lr\x18\x02 \x01(\x02H\x01\x88\x01\x01\x12\x17\n\nbatch_size\x18\x03 \x01(\x05H\x02\x88\x01\x01\x12\x18\n\x0b\x62uffer_size\x18\x04 \x01(\x05H\x03\x88\x01\x01\x12\x15\n\x08\x65pisodes\x18\x05 \x01(\x05H\x04\x88\x01\x01\x12\x12\n\x05gamma\x18\x06 \x01(\x02H\x05\x88\x01\x01\x12\x14\n\x07\x65psilon\x18\x07 \x01(\x02H\x06\x88\x01\x01\x12!\n\x14\x63ongestion_threshold\x18\x08 \x01(\x05H\x07\x88\x01\x01\x12\x1f\n\x12num_virtual_agents\x18\t \x01(\x05H\x08\x88\x01\x01\x12\x18\n\x0breward_exit\x18\n \x01(\x02H\t\x88\x01\x01\x12\x18\n\x0breward_fire\x18\x0b \x01(\x02H\n\x88\x01\x01\x12\x1b\n\x0ereward_invalid\x18\x0c \x01(\x02H\x0b\x88\x01\x01\x12\x19\n\x0creward_valid\x18\r \x01(\x02H\x0c\x88\x01\x01\x12\x1e\n\x11reward_congestion\x18\x0e \x01(\x02H\r\x88\x01\x01\x42\x07\n\x05_betaB\x05\n\x03_lrB\r\n\x0b_batch_sizeB\x0e\n\x0c_buffer_sizeB\x0b\n\t_episodesB\x08\n\x06_gammaB\n\n\x08_epsilonB\x17\n\x15_congestion_thresholdB\x15\n\x13_num_virtual_agentsB\x0e\n\x0c_reward_exitB\x0e\n\x0c_reward_fireB\x11\n\x0f_reward_invalidB\x0f\n\r_reward_validB\x14\n\x12_reward_congestion2\xca\x01\n\x15ReinforcementLearning\x12\x31\n\nTrainModel\x12\x10.rl.TrainRequest\x1a\x11.rl.TrainResponse\x12\x38\n\tInference\x12\x14.rl.InferenceRequest\x1a\x15.rl.InferenceResponse\x12\x44\n\rGetRewardPath\x12\x18.rl.GetRewardPathRequest\x1a\x19.rl.GetRewardPathResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.grpc_server.evacuai_rl_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_TRAINREQUEST']._serialized_start=41
  _globals['_TRAINREQUEST']._serialized_end=185
  _globals['_TRAINRESPONSE']._serialized_start=187
  _globals['_TRAINRESPONSE']._serialized_end=220
  _globals['_INFERENCEREQUEST']._serialized_start=222
  _globals['_INFERENCEREQUEST']._serialized_end=347
  _globals['_GETREWARDPATHREQUEST']._serialized_start=349
  _globals['_GETREWARDPATHREQUEST']._serialized_end=468
  _globals['_PATH']._serialized_start=470
  _globals['_PATH']._serialized_end=528
  _globals['_INFERENCERESPONSE']._serialized_start=530
  _globals['_INFERENCERESPONSE']._serialized_end=580
  _globals['_GETREWARDPATHRESPONSE']._serialized_start=582
  _globals['_GETREWARDPATHRESPONSE']._serialized_end=621
  _globals['_HYPPARAMS']._serialized_start=624
  _globals['_HYPPARAMS']._serialized_end=1215
  _globals['_REINFORCEMENTLEARNING']._serialized_start=1218
  _globals['_REINFORCEMENTLEARNING']._serialized_end=1420
# @@protoc_insertion_point(module_scope)
