	O??e?c??O??e?c??!O??e?c??	??wF/U @??wF/U @!??wF/U @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$O??e?c??u????A??^)??Y333333??*	333333R@2F
Iterator::Model??@??ǘ?!?????@@)??~j?t??1?Q?Q:@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ǘ????!?@?@6@)??ǘ????1?@?@6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?|a2U??!?ʨ???E@)2U0*???1???5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?I+???!?C8?C8.@)?J?4??1;?;?'@:Preprocessing2U
Iterator::Model::ParallelMapV2??_?Lu?!$I?$I?@)??_?Lu?1$I?$I?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??+e???!?p??P@)/n??r?1?-?-@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?Le?!$I?$I?@)??_?Le?1$I?$I?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(??0??!UFeTF?0@)??_?LU?1$I?$I???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t13.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??wF/U @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	u????u????!u????      ??!       "      ??!       *      ??!       2	??^)????^)??!??^)??:      ??!       B      ??!       J	333333??333333??!333333??R      ??!       Z	333333??333333??!333333??JCPU_ONLYY??wF/U @b 