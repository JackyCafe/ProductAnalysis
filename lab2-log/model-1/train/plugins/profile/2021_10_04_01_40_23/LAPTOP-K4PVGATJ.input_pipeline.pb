	?:pΈ????:pΈ???!?:pΈ???	6Ӈ??+@6Ӈ??+@!6Ӈ??+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?:pΈ???1?Zd??A??????Y?(??0??*	?????Q@2F
Iterator::Model??B?iޡ?!/?????I@)?X?? ??1???z
kE@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX?5?;N??!R[Im%?8@)y?&1???1|??w4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA??ǘ???!????b:0@)?? ?rh??1??????(@:Preprocessing2U
Iterator::Model::ParallelMapV2Ǻ???v?!???_ @)Ǻ???v?1???_ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???x?&??!?_@}H@)"??u??q?1VR[Im%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!XOa=??@)?????g?1XOa=??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?Le?!??2??h@)??_?Le?1??2??h@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(??0??!???>??1@)a2U0*?S?1?}???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 13.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t15.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.95Ӈ??+@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	1?Zd??1?Zd??!1?Zd??      ??!       "      ??!       *      ??!       2	????????????!??????:      ??!       B      ??!       J	?(??0???(??0??!?(??0??R      ??!       Z	?(??0???(??0??!?(??0??JCPU_ONLYY5Ӈ??+@b 