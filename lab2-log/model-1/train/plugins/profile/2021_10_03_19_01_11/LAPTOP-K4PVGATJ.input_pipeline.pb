	f?c]?F??f?c]?F??!f?c]?F??	??f??@??f??@!??f??@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$f?c]?F????ZӼ???Aڬ?\m???Y{?G?z??*	     ?Q@2F
Iterator::Model?
F%u??!e?v?'?A@)????<,??1??K=?;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatA??ǘ???!?h?́D?@)"??u????16??98@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???QI??!?D+l$4@)/?$???1z2~?ԓ-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?+e?X??!΁D+lP@)???_vO~?1??V?$@:Preprocessing2U
Iterator::Model::ParallelMapV2?+e?Xw?!΁D+l @)?+e?Xw?1΁D+l @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!i?́D+@){?G?zt?1i?́D+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!??9?h@)ŏ1w-!o?1??9?h@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Mb??!!?
??6@)_?Q?[?1?d?v?'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t15.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??f??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ZӼ?????ZӼ???!??ZӼ???      ??!       "      ??!       *      ??!       2	ڬ?\m???ڬ?\m???!ڬ?\m???:      ??!       B      ??!       J	{?G?z??{?G?z??!{?G?z??R      ??!       Z	{?G?z??{?G?z??!{?G?z??JCPU_ONLYY??f??@b 