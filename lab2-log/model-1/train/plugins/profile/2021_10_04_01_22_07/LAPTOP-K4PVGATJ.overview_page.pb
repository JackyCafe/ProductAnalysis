?	jM??St??jM??St??!jM??St??	G???7&@G???7&@!G???7&@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$jM??St??B>?٬???A??Q???Y?V-??*	?????YK@2F
Iterator::Model46<?R??!Q??G?C@)???????1K????B?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?St$????!??,??X>@)y?&1???1NSZ5?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<?R??!Q??G?3@)??ǘ????1??G???-@:Preprocessing2U
Iterator::Model::ParallelMapV2U???N@s?!??D/!@)U???N@s?1??D/!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip*??Dؠ?!??e?N@)????Mbp?1~???@@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceǺ???f?!r??y@)Ǻ???f?1r??y@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!?iJ?F@)??_?Le?1?iJ?F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(??0??!??m}|6@)Ǻ???V?1r??y@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9G???7&@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B>?٬???B>?٬???!B>?٬???      ??!       "      ??!       *      ??!       2	??Q?????Q???!??Q???:      ??!       B      ??!       J	?V-???V-??!?V-??R      ??!       Z	?V-???V-??!?V-??JCPU_ONLYYG???7&@b Y      Y@q?X?'??S@"?	
both?Your program is MODERATELY input-bound because 11.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t17.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?79.1197% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 