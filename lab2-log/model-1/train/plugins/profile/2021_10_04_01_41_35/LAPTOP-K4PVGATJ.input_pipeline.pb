	M?O????M?O????!M?O????	????P%@????P%@!????P%@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$M?O???????Mb??A?U??????Y䃞ͪϥ?*	33333?P@2F
Iterator::Model???QI??!?v??hE@)$????ۗ?1?b??.A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??d?`T??!<k???f:@)2??%䃎?1W?m??5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?W[?????!?WĔyE6@)?(??0??1?Co?$2@:Preprocessing2U
Iterator::Model::ParallelMapV2?g??s?u?!?KtB?D@)?g??s?u?1?KtB?D@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipn????!q?N??L@)??ZӼ?t?1{ja??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!?/4??@)?~j?t?h?1?/4??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceǺ???f?!?N???@)Ǻ???f?1?N???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ǘ????!M?l??7@)/n??R?1?V?m???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t11.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9????P%@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Mb?????Mb??!???Mb??      ??!       "      ??!       *      ??!       2	?U???????U??????!?U??????:      ??!       B      ??!       J	䃞ͪϥ?䃞ͪϥ?!䃞ͪϥ?R      ??!       Z	䃞ͪϥ?䃞ͪϥ?!䃞ͪϥ?JCPU_ONLYY????P%@b 