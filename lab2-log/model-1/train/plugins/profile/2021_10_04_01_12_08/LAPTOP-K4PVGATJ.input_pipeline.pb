	?s??????s?????!?s?????	??q??"@??q??"@!??q??"@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?s??????(??0??AV}??b??Y?J?4??*	     @R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatjM????!?B?
*:@)??ܵ?|??1???6@:Preprocessing2F
Iterator::Model8??d?`??!*T?P?B;@)? ?	???1?#F?5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapu????!?W?^?zD@)???_vO??1bĈ#F4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?W[?????!??իW?4@)a??+e??1ȏ?~?0@:Preprocessing2U
Iterator::Model::ParallelMapV2;?O??nr?!?
*T?@);?O??nr?1?
*T?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???V?/??!??իW/R@)	?^)?p?1t?Ν;w@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!8p@)?~j?t?h?18p@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vOf?!mٲe˖@)??_vOf?1mٲe˖@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t13.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??q??"@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(??0???(??0??!?(??0??      ??!       "      ??!       *      ??!       2	V}??b??V}??b??!V}??b??:      ??!       B      ??!       J	?J?4???J?4??!?J?4??R      ??!       Z	?J?4???J?4??!?J?4??JCPU_ONLYY??q??"@b 