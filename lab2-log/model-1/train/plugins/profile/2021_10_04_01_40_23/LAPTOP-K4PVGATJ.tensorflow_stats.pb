"?G
uHostFlushSummaryWriter"FlushSummaryWriter(1?????L?@9?????L?@A?????L?@I?????L?@a???i???i???i????Unknown?
BHostIDLE"IDLE133333G?@A33333G?@aٳʑ????i?-??8???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(133333sM@933333sM@A33333sM@I33333sM@aS
/w???i<Wi?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1ffffffG@9ffffffG@AffffffG@IffffffG@a&Ѕw?a??i}nG????Unknown
dHostDataset"Iterator::Model(133333sA@933333sA@A?????L=@I?????L=@a?Ћ???i?_?]???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1ffffff5@9ffffff5@Affffff5@Iffffff5@a;'?26w?i]M$?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1fffff?0@9fffff?0@A      ,@I      ,@a',?^n?i*?R??????Unknown
iHostWriteSummary"WriteSummary(1      %@9      %@A      %@I      %@aD?C!?f?iGj??????Unknown?
?	HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1333333!@9333333!@A333333!@I333333!@a???b?iI1?U????Unknown
?
HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1333333&@9333333&@A      !@I      !@a?#Cppb?i<U???????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?????? @9?????? @A?????? @I?????? @aN?ݏ^b?i3???????Unknown
^HostGatherV2"GatherV2(1333333@9333333@A333333@I333333@ad럞??]?i?U>????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1??????@9??????@A??????@I??????@a?8?jI?Z?i? c????Unknown
gHostStridedSlice"strided_slice(1??????@9??????@A??????@I??????@a?(Q?Y?i?3?? ???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff@9ffffff@Affffff@Iffffff@a??V7?KX?iV@???,???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?ұu?U?i?Q??7???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?@@9     ?@@A333333@I333333@a???R?i?|ߓA???Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a?lh??|P?i??-?WI???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1ffffff@9ffffff@Affffff@Iffffff@a?lh??|P?i,?{4?Q???Unknown
[HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a$O"??P?iTv???Y???Unknown
YHostPow"Adam/Pow(1      @9      @A      @I      @a??n&J?i?q`???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333??A333333@I333333??a>???*I?i?
?if???Unknown
eHost
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a??V7?KH?id??}|l???Unknown?
THostSub"sub(1ffffff@9ffffff@Affffff@Iffffff@a??V7?KH?i??u?r???Unknown
\HostGreater"Greater(1??????@9??????@A??????@I??????@a?I?лmG?i????jx???Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1??????@9??????@A??????@I??????@a?I?лmG?i)?SF~???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a\>j??F?i???9?????Unknown
VHostSum"Sum_3(1??????@9??????@A??????@I??????@a\>j??F?i1:??????Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1ffffff@9ffffff@Affffff@Iffffff@ax[?6.?C?i??_k?????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff??9ffffff??Affffff??Iffffff??a?lh??|@?i?????????Unknown
iHostMean"mean_squared_error/Mean(1ffffff??9ffffff??Affffff??Iffffff??a?lh??|@?i???ɖ???Unknown
X HostEqual"Equal(1????????9????????A????????I????????a?b?k?<??iʋ?Z?????Unknown
]!HostCast"Adam/Cast_1(1333333??9333333??A333333??I333333??ad럞??=?i?_oqa????Unknown
u"HostSum"$mean_squared_error/weighted_loss/Sum(1333333??9333333??A333333??I333333??ad럞??=?i?3??????Unknown
t#HostReadVariableOp"Adam/Cast/ReadVariableOp(1????????9????????A????????I????????a"t??l?;?i?d}?????Unknown
t$HostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a??n&:?i??=˨???Unknown
V%HostCast"Cast(1      ??9      ??A      ??I      ??a??n&:?is??????Unknown
X&HostCast"Cast_5(1      ??9      ??A      ??I      ??a??n&:?iS?#M????Unknown
?'HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a??n&:?i3?(?????Unknown
?(HostSquaredDifference"$mean_squared_error/SquaredDifference(1      ??9      ??A      ??I      ??a??n&:?i*@-ϵ???Unknown
~)HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????????9????????A????????I????????a\>j??6?i?qM ?????Unknown
u*HostSum"$gradient_tape/mean_squared_error/Sum(1????????9????????A????????I????????a\>j??6?i??Zs????Unknown
}+HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a\>j??6?iYhE????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_1(1333333??9333333??A333333??I333333??a?%?Q?4?i??p?????Unknown
?-HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????(@9??????(@A333333??I333333??a?%?Q?4?i?J??y????Unknown
`.HostDivNoNan"
div_no_nan(1333333??9333333??A333333??I333333??a?%?Q?4?ir?E????Unknown
u/HostReadVariableOp"div_no_nan/ReadVariableOp(1333333??9333333??A333333??I333333??a?%?Q?4?i%?6??????Unknown
T0HostAbs"Abs(1????????9????????A????????I????????a??
3?iɕ??????Unknown
o1HostReadVariableOp"Adam/ReadVariableOp(1????????9????????A????????I????????a??
3?im??qt????Unknown
T2HostMul"Mul(1????????9????????A????????I????????a??
3?i?DS?????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_5(1      ??9      ??A      ??I      ??a????Z1?i??ī????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_6(1      ??9      ??A      ??I      ??a????Z1?i;VE.????Unknown
b5HostDivNoNan"div_no_nan_2(1      ??9      ??A      ??I      ??a????Z1?iд?\Y????Unknown
}6HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      ??9      ??A      ??I      ??a????Z1?ieF??????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_3(1????????9????????A????????I????????a?b?k?</?i????x????Unknown
V8HostMean"Mean(1????????9????????A????????I????????a?b?k?</?iq??Tl????Unknown
X9HostMean"Mean_1(1????????9????????A????????I????????a?b?k?</?i?E:$`????Unknown
V:HostSum"Sum_2(1????????9????????A????????I????????a?b?k?</?i}??S????Unknown
?;HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1????????9????????A????????I????????a?b?k?</?i???G????Unknown
v<HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????a"t??l?+?iz?T
????Unknown
[=HostPow"
Adam/Pow_1(1????????9????????A????????I????????a"t??l?+?i??!Q?????Unknown
X>HostCast"Cast_2(1????????9????????A????????I????????a"t??l?+?ih??|????Unknown
u?HostMul"$gradient_tape/mean_squared_error/Mul(1????????9????????A????????I????????a"t??l?+?i???8????Unknown
@HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1????????9????????A????????I????????a"t??l?+?iV7?%?????Unknown
uAHostSub"$gradient_tape/mean_squared_error/sub(1????????9????????A????????I????????a"t??l?+?i?OVl?????Unknown
vBHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a??V7?K(?i5?I*6????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff??9ffffff??Affffff??Iffffff??a??V7?K(?i?:=??????Unknown
XDHostCast"Cast_7(1ffffff??9ffffff??Affffff??Iffffff??a??V7?K(?i?0??????Unknown
bEHostDivNoNan"div_no_nan_1(1ffffff??9ffffff??Affffff??Iffffff??a??V7?K(?im%$d?????Unknown
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a??V7?K(?i՚"I????Unknown
?GHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1ffffff??9ffffff??Affffff??Iffffff??a??V7?K(?i=??????Unknown
XHHostCast"Cast_6(1333333??9333333??A333333??I333333??a?%?Q?$?i??$????Unknown
wIHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?%?Q?$?i??>Jh????Unknown
yJHostReadVariableOp"div_no_nan_2/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?%?Q?$?iH?X?????Unknown
wKHostCast"%gradient_tape/mean_squared_error/Cast(1333333??9333333??A333333??I333333??a?%?Q?$?i?Yr?????Unknown
wLHostMul"&gradient_tape/mean_squared_error/mul_1(1333333??9333333??A333333??I333333??a?%?Q?$?i?+??O????Unknown
|MHostDivNoNan"&mean_squared_error/weighted_loss/value(1333333??9333333??A333333??I333333??a?%?Q?$?iS???????Unknown
wNHostReadVariableOp"div_no_nan_2/ReadVariableOp(1      ??9      ??A      ??I      ??a????Z!?i?-?ʲ????Unknown
aOHostIdentity"Identity(1333333??9333333??A333333??I333333??a?%?Q??i?seY????Unknown?
yPHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?%?Q??i?????????Unknown*?G
uHostFlushSummaryWriter"FlushSummaryWriter(1?????L?@9?????L?@A?????L?@I?????L?@a??J???i??J????Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(133333sM@933333sM@A33333sM@I33333sM@a??SU˕??i???e4????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1ffffffG@9ffffffG@AffffffG@IffffffG@a??n????i?????U???Unknown
dHostDataset"Iterator::Model(133333sA@933333sA@A?????L=@I?????L=@a=?\lXx??iDmZ??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1ffffff5@9ffffff5@Affffff5@Iffffff5@a?O?)_i??i?? tK????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1fffff?0@9fffff?0@A      ,@I      ,@a?W4yu?i2?h?=???Unknown
iHostWriteSummary"WriteSummary(1      %@9      %@A      %@I      %@a?g?p?i5?6as<???Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1333333!@9333333!@A333333!@I333333!@ac8&]?aj?im????V???Unknown
?	HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1333333&@9333333&@A      !@I      !@aVꓚj?iWy.?p???Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?????? @9?????? @A?????? @I?????? @a<Novi?i??C^????Unknown
^HostGatherV2"GatherV2(1333333@9333333@A333333@I333333@ax?ܮ?d?i`??#:????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1??????@9??????@A??????@I??????@a*?n?c?iG4?????Unknown
gHostStridedSlice"strided_slice(1??????@9??????@A??????@I??????@a???Ib?iD??NX????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff@9ffffff@Affffff@Iffffff@a???-a?iW?y?????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a}&?^?i??z??????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?@@9     ?@@A333333@I333333@ac8&]?aZ?i?)S????Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a?+o?MPW?iH?
z?????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1ffffff@9ffffff@Affffff@Iffffff@a?+o?MPW?iނ??]	???Unknown
[HostAddV2"Adam/add(1??????@9??????@A??????@I??????@aƏJ>>?V?i&?@????Unknown
YHostPow"Adam/Pow(1      @9      @A      @I      @aKJ??gR?iL??(????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1333333@9333333??A333333@I333333??a??%??Q?i#`???&???Unknown
eHost
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a???-Q?i?`+cg/???Unknown?
THostSub"sub(1ffffff@9ffffff@Affffff@Iffffff@a???-Q?i5as<?7???Unknown
\HostGreater"Greater(1??????@9??????@A??????@I??????@a?v?
??P?ip???F@???Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1??????@9??????@A??????@I??????@a?v?
??P?i?=~ߎH???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@aN?o'?O?i?A??P???Unknown
VHostSum"Sum_3(1??????@9??????@A??????@I??????@aN?o'?O?i??s?X???Unknown
?HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1ffffff@9ffffff@Affffff@Iffffff@a????8L?i???_???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff??9ffffff??Affffff??Iffffff??a?+o?MPG?iS???de???Unknown
iHostMean"mean_squared_error/Mean(1ffffff??9ffffff??Affffff??Iffffff??a?+o?MPG?i?`?8k???Unknown
XHostEqual"Equal(1????????9????????A????????I????????a??%?.F?i?X?p???Unknown
] HostCast"Adam/Cast_1(1333333??9333333??A333333??I333333??ax?ܮ?D?i???[?u???Unknown
u!HostSum"$mean_squared_error/weighted_loss/Sum(1333333??9333333??A333333??I333333??ax?ܮ?D?i???_,{???Unknown
t"HostReadVariableOp"Adam/Cast/ReadVariableOp(1????????9????????A????????I????????aD?????C?iڮ?????Unknown
t#HostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??aKJ??gB?imAvЮ????Unknown
V$HostCast"Cast(1      ??9      ??A      ??I      ??aKJ??gB?i ???H????Unknown
X%HostCast"Cast_5(1      ??9      ??A      ??I      ??aKJ??gB?i?fC??????Unknown
?&HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??aKJ??gB?i&???|????Unknown
?'HostSquaredDifference"$mean_squared_error/SquaredDifference(1      ??9      ??A      ??I      ??aKJ??gB?i???????Unknown
~(HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1????????9????????A????????I????????aN?o'???i????????Unknown
u)HostSum"$gradient_tape/mean_squared_error/Sum(1????????9????????A????????I????????aN?o'???i?g?k????Unknown
}*HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????aN?o'???i?մP????Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_1(1333333??9333333??A333333??I333333??a?D???r=?iG?ӭ?????Unknown
?,HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1??????(@9??????(@A333333??I333333??a?D???r=?i???
j????Unknown
`-HostDivNoNan"
div_no_nan(1333333??9333333??A333333??I333333??a?D???r=?i?hh????Unknown
u.HostReadVariableOp"div_no_nan/ReadVariableOp(1333333??9333333??A333333??I333333??a?D???r=?iBD0?Ʊ???Unknown
T/HostAbs"Abs(1????????9????????A????????I????????a~?J???:?i????&????Unknown
o0HostReadVariableOp"Adam/ReadVariableOp(1????????9????????A????????I????????a~?J???:?i???o?????Unknown
T1HostMul"Mul(1????????9????????A????????I????????a~?J???:?iS EE?????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_5(1      ??9      ??A      ??I      ??ad??l?8?i`?ޒ?????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_6(1      ??9      ??A      ??I      ??ad??l?8?im?x?????Unknown
b4HostDivNoNan"div_no_nan_2(1      ??9      ??A      ??I      ??ad??l?8?izE.????Unknown
}5HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      ??9      ??A      ??I      ??ad??l?8?i???{+????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_3(1????????9????????A????????I????????a??%?.6?iE!?A?????Unknown
V7HostMean"Mean(1????????9????????A????????I????????a??%?.6?iFZ?????Unknown
X8HostMean"Mean_1(1????????9????????A????????I????????a??%?.6?i?j1?s????Unknown
V9HostSum"Sum_2(1????????9????????A????????I????????a??%?.6?i??6????Unknown
?:HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1????????9????????A????????I????????a??%?.6?i=??X?????Unknown
v;HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????aD?????3?i?F??m????Unknown
[<HostPow"
Adam/Pow_1(1????????9????????A????????I????????aD?????3?i???????Unknown
X=HostCast"Cast_2(1????????9????????A????????I????????aD?????3?i?kV????Unknown
u>HostMul"$gradient_tape/mean_squared_error/Mul(1????????9????????A????????I????????aD?????3?i??1Q?????Unknown
?HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1????????9????????A????????I????????aD?????3?im?F?>????Unknown
u@HostSub"$gradient_tape/mean_squared_error/sub(1????????9????????A????????I????????aD?????3?i?"[Ͳ????Unknown
vAHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a???-1?i?"???????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff??9ffffff??Affffff??Iffffff??a???-1?i!#?9?????Unknown
XCHostCast"Cast_7(1ffffff??9ffffff??Affffff??Iffffff??a???-1?iC#Q?#????Unknown
bDHostDivNoNan"div_no_nan_1(1ffffff??9ffffff??Affffff??Iffffff??a???-1?ie#??I????Unknown
wEHostReadVariableOp"div_no_nan_1/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a???-1?i?#?\o????Unknown
?FHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1ffffff??9ffffff??Affffff??Iffffff??a???-1?i?#G?????Unknown
XGHostCast"Cast_6(1333333??9333333??A333333??I333333??a?D???r-?i}??Al????Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?D???r-?iQ?epC????Unknown
yIHostReadVariableOp"div_no_nan_2/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?D???r-?i%m??????Unknown
wJHostCast"%gradient_tape/mean_squared_error/Cast(1333333??9333333??A333333??I333333??a?D???r-?i?ڄ??????Unknown
wKHostMul"&gradient_tape/mean_squared_error/mul_1(1333333??9333333??A333333??I333333??a?D???r-?i?H??????Unknown
|LHostDivNoNan"&mean_squared_error/weighted_loss/value(1333333??9333333??A333333??I333333??a?D???r-?i???*?????Unknown
wMHostReadVariableOp"div_no_nan_2/ReadVariableOp(1      ??9      ??A      ??I      ??ad??l?(?i'?p?(????Unknown
aNHostIdentity"Identity(1333333??9333333??A333333??I333333??a?D???r?iI?h????Unknown?
yOHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333??9333333??A333333??I333333??a?D???r?i?????????Unknown