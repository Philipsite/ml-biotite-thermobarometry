٧
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
n
ConstConst*
_output_shapes

:*
dtype0*1
value(B&"K	K<�\�9�O=�e�=�=N'�8
p
Const_1Const*
_output_shapes

:*
dtype0*1
value(B&"X.@�=�4�?���?n��?VUX<
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
count_2VarHandleOp*
_output_shapes
: *

debug_name
count_2/*
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
|
total_2VarHandleOp*
_output_shapes
: *

debug_name
total_2/*
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
|
count_3VarHandleOp*
_output_shapes
: *

debug_name
count_3/*
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
|
total_3VarHandleOp*
_output_shapes
: *

debug_name
total_3/*
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
|
count_4VarHandleOp*
_output_shapes
: *

debug_name
count_4/*
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
|
total_4VarHandleOp*
_output_shapes
: *

debug_name
total_4/*
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
�
Adam/v/dense_71/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_71/bias/*
dtype0*
shape:*%
shared_nameAdam/v/dense_71/bias
y
(Adam/v/dense_71/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_71/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_71/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_71/bias/*
dtype0*
shape:*%
shared_nameAdam/m/dense_71/bias
y
(Adam/m/dense_71/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_71/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_71/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_71/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/v/dense_71/kernel
�
*Adam/v/dense_71/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_71/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_71/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_71/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/m/dense_71/kernel
�
*Adam/m/dense_71/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_71/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_70/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_70/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_70/bias
z
(Adam/v/dense_70/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_70/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_70/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_70/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_70/bias
z
(Adam/m/dense_70/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_70/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_70/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_70/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_70/kernel
�
*Adam/v/dense_70/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_70/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_70/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_70/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_70/kernel
�
*Adam/m/dense_70/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_70/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_69/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_69/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_69/bias
z
(Adam/v/dense_69/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_69/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_69/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_69/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_69/bias
z
(Adam/m/dense_69/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_69/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_69/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_69/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_69/kernel
�
*Adam/v/dense_69/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_69/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_69/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_69/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_69/kernel
�
*Adam/m/dense_69/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_69/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_68/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_68/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_68/bias
z
(Adam/v/dense_68/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_68/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_68/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_68/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_68/bias
z
(Adam/m/dense_68/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_68/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_68/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_68/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_68/kernel
�
*Adam/v/dense_68/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_68/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_68/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_68/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_68/kernel
�
*Adam/m/dense_68/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_68/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_67/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_67/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_67/bias
z
(Adam/v/dense_67/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_67/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_67/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_67/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_67/bias
z
(Adam/m/dense_67/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_67/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_67/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_67/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_67/kernel
�
*Adam/v/dense_67/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_67/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_67/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_67/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_67/kernel
�
*Adam/m/dense_67/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_67/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_66/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_66/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_66/bias
z
(Adam/v/dense_66/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_66/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_66/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_66/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_66/bias
z
(Adam/m/dense_66/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_66/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_66/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_66/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_66/kernel
�
*Adam/v/dense_66/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_66/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_66/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_66/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_66/kernel
�
*Adam/m/dense_66/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_66/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_65/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_65/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_65/bias
z
(Adam/v/dense_65/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_65/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_65/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_65/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_65/bias
z
(Adam/m/dense_65/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_65/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_65/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_65/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/v/dense_65/kernel
�
*Adam/v/dense_65/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_65/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_65/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_65/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/m/dense_65/kernel
�
*Adam/m/dense_65/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_65/kernel*
_output_shapes
:	�*
dtype0
�
current_learning_rateVarHandleOp*
_output_shapes
: *&

debug_namecurrent_learning_rate/*
dtype0*
shape: *&
shared_namecurrent_learning_rate
w
)current_learning_rate/Read/ReadVariableOpReadVariableOpcurrent_learning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_71/biasVarHandleOp*
_output_shapes
: *

debug_namedense_71/bias/*
dtype0*
shape:*
shared_namedense_71/bias
k
!dense_71/bias/Read/ReadVariableOpReadVariableOpdense_71/bias*
_output_shapes
:*
dtype0
�
dense_71/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_71/kernel/*
dtype0*
shape:	�* 
shared_namedense_71/kernel
t
#dense_71/kernel/Read/ReadVariableOpReadVariableOpdense_71/kernel*
_output_shapes
:	�*
dtype0
�
dense_70/biasVarHandleOp*
_output_shapes
: *

debug_namedense_70/bias/*
dtype0*
shape:�*
shared_namedense_70/bias
l
!dense_70/bias/Read/ReadVariableOpReadVariableOpdense_70/bias*
_output_shapes	
:�*
dtype0
�
dense_70/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_70/kernel/*
dtype0*
shape:
��* 
shared_namedense_70/kernel
u
#dense_70/kernel/Read/ReadVariableOpReadVariableOpdense_70/kernel* 
_output_shapes
:
��*
dtype0
�
dense_69/biasVarHandleOp*
_output_shapes
: *

debug_namedense_69/bias/*
dtype0*
shape:�*
shared_namedense_69/bias
l
!dense_69/bias/Read/ReadVariableOpReadVariableOpdense_69/bias*
_output_shapes	
:�*
dtype0
�
dense_69/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_69/kernel/*
dtype0*
shape:
��* 
shared_namedense_69/kernel
u
#dense_69/kernel/Read/ReadVariableOpReadVariableOpdense_69/kernel* 
_output_shapes
:
��*
dtype0
�
dense_68/biasVarHandleOp*
_output_shapes
: *

debug_namedense_68/bias/*
dtype0*
shape:�*
shared_namedense_68/bias
l
!dense_68/bias/Read/ReadVariableOpReadVariableOpdense_68/bias*
_output_shapes	
:�*
dtype0
�
dense_68/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_68/kernel/*
dtype0*
shape:
��* 
shared_namedense_68/kernel
u
#dense_68/kernel/Read/ReadVariableOpReadVariableOpdense_68/kernel* 
_output_shapes
:
��*
dtype0
�
dense_67/biasVarHandleOp*
_output_shapes
: *

debug_namedense_67/bias/*
dtype0*
shape:�*
shared_namedense_67/bias
l
!dense_67/bias/Read/ReadVariableOpReadVariableOpdense_67/bias*
_output_shapes	
:�*
dtype0
�
dense_67/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_67/kernel/*
dtype0*
shape:
��* 
shared_namedense_67/kernel
u
#dense_67/kernel/Read/ReadVariableOpReadVariableOpdense_67/kernel* 
_output_shapes
:
��*
dtype0
�
dense_66/biasVarHandleOp*
_output_shapes
: *

debug_namedense_66/bias/*
dtype0*
shape:�*
shared_namedense_66/bias
l
!dense_66/bias/Read/ReadVariableOpReadVariableOpdense_66/bias*
_output_shapes	
:�*
dtype0
�
dense_66/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_66/kernel/*
dtype0*
shape:
��* 
shared_namedense_66/kernel
u
#dense_66/kernel/Read/ReadVariableOpReadVariableOpdense_66/kernel* 
_output_shapes
:
��*
dtype0
�
dense_65/biasVarHandleOp*
_output_shapes
: *

debug_namedense_65/bias/*
dtype0*
shape:�*
shared_namedense_65/bias
l
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
_output_shapes	
:�*
dtype0
�
dense_65/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_65/kernel/*
dtype0*
shape:	�* 
shared_namedense_65/kernel
t
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel*
_output_shapes
:	�*
dtype0
|
count_5VarHandleOp*
_output_shapes
: *

debug_name
count_5/*
dtype0	*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0	
�
varianceVarHandleOp*
_output_shapes
: *

debug_name	variance/*
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
w
meanVarHandleOp*
_output_shapes
: *

debug_namemean/*
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
�
#serving_default_normalization_inputPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConst_1Constdense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/biasdense_70/kerneldense_70/biasdense_71/kerneldense_71/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_21798513

NoOpNoOp
�d
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�d
value�dB�d B�d
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias*
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
�
0
1
2
!3
"4
)5
*6
17
28
99
:10
A11
B12
I13
J14
Q15
R16*
j
!0
"1
)2
*3
14
25
96
:7
A8
B9
I10
J11
Q12
R13*
X
S0
T1
U2
V3
W4
X5
Y6
Z7
[8
\9
]10
^11* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
 
h	capture_0
i	capture_1* 
�
j
_variables
k_iterations
l_current_learning_rate
m_index_dict
n
_momentums
o_velocities
p_update_step_xla*

qserving_default* 
* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_55layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*

rtrace_0* 

!0
"1*

!0
"1*

S0
T1* 
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

xtrace_0* 

ytrace_0* 
_Y
VARIABLE_VALUEdense_65/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_65/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*

U0
V1* 
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_66/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_66/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

10
21*

10
21*

W0
X1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_67/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_67/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*

Y0
Z1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_68/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_68/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*

[0
\1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_69/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_69/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*

]0
^1* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_70/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_70/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_71/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_71/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

0
1
2*
<
0
1
2
3
4
5
6
7*
,
�0
�1
�2
�3
�4*
* 
* 
 
h	capture_0
i	capture_1* 
 
h	capture_0
i	capture_1* 
 
h	capture_0
i	capture_1* 
 
h	capture_0
i	capture_1* 
* 
* 
�
k0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcurrent_learning_rate;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
* 
 
h	capture_0
i	capture_1* 
* 
* 
* 
* 

S0
T1* 
* 
* 
* 
* 
* 
* 

U0
V1* 
* 
* 
* 
* 
* 
* 

W0
X1* 
* 
* 
* 
* 
* 
* 

Y0
Z1* 
* 
* 
* 
* 
* 
* 

[0
\1* 
* 
* 
* 
* 
* 
* 

]0
^1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_65/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_65/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_65/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_65/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_66/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_66/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_66/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_66/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_67/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_67/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_67/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_67/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_68/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_68/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_68/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_68/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_69/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_69/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_69/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_69/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_70/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_70/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_70/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_70/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_71/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_71/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_71/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_71/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemeanvariancecount_5dense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/biasdense_70/kerneldense_70/biasdense_71/kerneldense_71/bias	iterationcurrent_learning_rateAdam/m/dense_65/kernelAdam/v/dense_65/kernelAdam/m/dense_65/biasAdam/v/dense_65/biasAdam/m/dense_66/kernelAdam/v/dense_66/kernelAdam/m/dense_66/biasAdam/v/dense_66/biasAdam/m/dense_67/kernelAdam/v/dense_67/kernelAdam/m/dense_67/biasAdam/v/dense_67/biasAdam/m/dense_68/kernelAdam/v/dense_68/kernelAdam/m/dense_68/biasAdam/v/dense_68/biasAdam/m/dense_69/kernelAdam/v/dense_69/kernelAdam/m/dense_69/biasAdam/v/dense_69/biasAdam/m/dense_70/kernelAdam/v/dense_70/kernelAdam/m/dense_70/biasAdam/v/dense_70/biasAdam/m/dense_71/kernelAdam/v/dense_71/kernelAdam/m/dense_71/biasAdam/v/dense_71/biastotal_4count_4total_3count_3total_2count_2total_1count_1totalcountConst_2*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_21799534
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_5dense_65/kerneldense_65/biasdense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/biasdense_69/kerneldense_69/biasdense_70/kerneldense_70/biasdense_71/kerneldense_71/bias	iterationcurrent_learning_rateAdam/m/dense_65/kernelAdam/v/dense_65/kernelAdam/m/dense_65/biasAdam/v/dense_65/biasAdam/m/dense_66/kernelAdam/v/dense_66/kernelAdam/m/dense_66/biasAdam/v/dense_66/biasAdam/m/dense_67/kernelAdam/v/dense_67/kernelAdam/m/dense_67/biasAdam/v/dense_67/biasAdam/m/dense_68/kernelAdam/v/dense_68/kernelAdam/m/dense_68/biasAdam/v/dense_68/biasAdam/m/dense_69/kernelAdam/v/dense_69/kernelAdam/m/dense_69/biasAdam/v/dense_69/biasAdam/m/dense_70/kernelAdam/v/dense_70/kernelAdam/m/dense_70/biasAdam/v/dense_70/biasAdam/m/dense_71/kernelAdam/v/dense_71/kernelAdam/m/dense_71/biasAdam/v/dense_71/biastotal_4count_4total_3count_3total_2count_2total_1count_1totalcount*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_21799714��
�,
�
F__inference_dense_67_layer_call_and_return_conditional_losses_21798807

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_67/bias/Regularizer/Abs/ReadVariableOp�/dense_67/bias/Regularizer/L2Loss/ReadVariableOp�.dense_67/kernel/Regularizer/Abs/ReadVariableOp�1dense_67/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_67/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_67/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_67/kernel/Regularizer/AbsAbs6dense_67/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_67/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_67/kernel/Regularizer/SumSum#dense_67/kernel/Regularizer/Abs:y:0,dense_67/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_67/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/kernel/Regularizer/mulMul*dense_67/kernel/Regularizer/mul/x:output:0(dense_67/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/kernel/Regularizer/addAddV2*dense_67/kernel/Regularizer/Const:output:0#dense_67/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_67/kernel/Regularizer/L2LossL2Loss9dense_67/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_67/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_67/kernel/Regularizer/mul_1Mul,dense_67/kernel/Regularizer/mul_1/x:output:0+dense_67/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_67/kernel/Regularizer/add_1AddV2#dense_67/kernel/Regularizer/add:z:0%dense_67/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_67/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_67/bias/Regularizer/AbsAbs4dense_67/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_67/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_67/bias/Regularizer/SumSum!dense_67/bias/Regularizer/Abs:y:0*dense_67/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/bias/Regularizer/mulMul(dense_67/bias/Regularizer/mul/x:output:0&dense_67/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/addAddV2(dense_67/bias/Regularizer/Const:output:0!dense_67/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_67/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_67/bias/Regularizer/L2LossL2Loss7dense_67/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_67/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_67/bias/Regularizer/mul_1Mul*dense_67/bias/Regularizer/mul_1/x:output:0)dense_67/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/add_1AddV2!dense_67/bias/Regularizer/add:z:0#dense_67/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_67/bias/Regularizer/Abs/ReadVariableOp0^dense_67/bias/Regularizer/L2Loss/ReadVariableOp/^dense_67/kernel/Regularizer/Abs/ReadVariableOp2^dense_67/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_67/bias/Regularizer/Abs/ReadVariableOp,dense_67/bias/Regularizer/Abs/ReadVariableOp2b
/dense_67/bias/Regularizer/L2Loss/ReadVariableOp/dense_67/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_67/kernel/Regularizer/Abs/ReadVariableOp.dense_67/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_6_21799083K
7dense_68_kernel_regularizer_abs_readvariableop_resource:
��
identity��.dense_68/kernel/Regularizer/Abs/ReadVariableOp�1dense_68/kernel/Regularizer/L2Loss/ReadVariableOpf
!dense_68/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_68/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_68_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_68/kernel/Regularizer/AbsAbs6dense_68/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_68/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_68/kernel/Regularizer/SumSum#dense_68/kernel/Regularizer/Abs:y:0,dense_68/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_68/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/kernel/Regularizer/mulMul*dense_68/kernel/Regularizer/mul/x:output:0(dense_68/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/kernel/Regularizer/addAddV2*dense_68/kernel/Regularizer/Const:output:0#dense_68/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_68_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_68/kernel/Regularizer/L2LossL2Loss9dense_68/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_68/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_68/kernel/Regularizer/mul_1Mul,dense_68/kernel/Regularizer/mul_1/x:output:0+dense_68/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_68/kernel/Regularizer/add_1AddV2#dense_68/kernel/Regularizer/add:z:0%dense_68/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_68/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^dense_68/kernel/Regularizer/Abs/ReadVariableOp2^dense_68/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_68/kernel/Regularizer/Abs/ReadVariableOp.dense_68/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
__inference_loss_fn_4_21799049K
7dense_67_kernel_regularizer_abs_readvariableop_resource:
��
identity��.dense_67/kernel/Regularizer/Abs/ReadVariableOp�1dense_67/kernel/Regularizer/L2Loss/ReadVariableOpf
!dense_67/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_67/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_67_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_67/kernel/Regularizer/AbsAbs6dense_67/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_67/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_67/kernel/Regularizer/SumSum#dense_67/kernel/Regularizer/Abs:y:0,dense_67/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_67/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/kernel/Regularizer/mulMul*dense_67/kernel/Regularizer/mul/x:output:0(dense_67/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/kernel/Regularizer/addAddV2*dense_67/kernel/Regularizer/Const:output:0#dense_67/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_67_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_67/kernel/Regularizer/L2LossL2Loss9dense_67/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_67/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_67/kernel/Regularizer/mul_1Mul,dense_67/kernel/Regularizer/mul_1/x:output:0+dense_67/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_67/kernel/Regularizer/add_1AddV2#dense_67/kernel/Regularizer/add:z:0%dense_67/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_67/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^dense_67/kernel/Regularizer/Abs/ReadVariableOp2^dense_67/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_67/kernel/Regularizer/Abs/ReadVariableOp.dense_67/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�,
�
F__inference_dense_69_layer_call_and_return_conditional_losses_21797760

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_69/bias/Regularizer/Abs/ReadVariableOp�/dense_69/bias/Regularizer/L2Loss/ReadVariableOp�.dense_69/kernel/Regularizer/Abs/ReadVariableOp�1dense_69/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_69/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_69/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_69/kernel/Regularizer/AbsAbs6dense_69/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_69/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_69/kernel/Regularizer/SumSum#dense_69/kernel/Regularizer/Abs:y:0,dense_69/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_69/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/kernel/Regularizer/mulMul*dense_69/kernel/Regularizer/mul/x:output:0(dense_69/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/kernel/Regularizer/addAddV2*dense_69/kernel/Regularizer/Const:output:0#dense_69/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_69/kernel/Regularizer/L2LossL2Loss9dense_69/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_69/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_69/kernel/Regularizer/mul_1Mul,dense_69/kernel/Regularizer/mul_1/x:output:0+dense_69/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_69/kernel/Regularizer/add_1AddV2#dense_69/kernel/Regularizer/add:z:0%dense_69/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_69/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_69/bias/Regularizer/AbsAbs4dense_69/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_69/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_69/bias/Regularizer/SumSum!dense_69/bias/Regularizer/Abs:y:0*dense_69/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/bias/Regularizer/mulMul(dense_69/bias/Regularizer/mul/x:output:0&dense_69/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/addAddV2(dense_69/bias/Regularizer/Const:output:0!dense_69/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_69/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_69/bias/Regularizer/L2LossL2Loss7dense_69/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_69/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_69/bias/Regularizer/mul_1Mul*dense_69/bias/Regularizer/mul_1/x:output:0)dense_69/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/add_1AddV2!dense_69/bias/Regularizer/add:z:0#dense_69/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_69/bias/Regularizer/Abs/ReadVariableOp0^dense_69/bias/Regularizer/L2Loss/ReadVariableOp/^dense_69/kernel/Regularizer/Abs/ReadVariableOp2^dense_69/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_69/bias/Regularizer/Abs/ReadVariableOp,dense_69/bias/Regularizer/Abs/ReadVariableOp2b
/dense_69/bias/Regularizer/L2Loss/ReadVariableOp/dense_69/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_69/kernel/Regularizer/Abs/ReadVariableOp.dense_69/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_sequential_13_layer_call_fn_21798219
normalization_input
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_21797980o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798215:($
"
_user_specified_name
21798213:($
"
_user_specified_name
21798211:($
"
_user_specified_name
21798209:($
"
_user_specified_name
21798207:($
"
_user_specified_name
21798205:(
$
"
_user_specified_name
21798203:(	$
"
_user_specified_name
21798201:($
"
_user_specified_name
21798199:($
"
_user_specified_name
21798197:($
"
_user_specified_name
21798195:($
"
_user_specified_name
21798193:($
"
_user_specified_name
21798191:($
"
_user_specified_name
21798189:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
__inference_loss_fn_7_21799100D
5dense_68_bias_regularizer_abs_readvariableop_resource:	�
identity��,dense_68/bias/Regularizer/Abs/ReadVariableOp�/dense_68/bias/Regularizer/L2Loss/ReadVariableOpd
dense_68/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_68/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_68_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_68/bias/Regularizer/AbsAbs4dense_68/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_68/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_68/bias/Regularizer/SumSum!dense_68/bias/Regularizer/Abs:y:0*dense_68/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/bias/Regularizer/mulMul(dense_68/bias/Regularizer/mul/x:output:0&dense_68/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/addAddV2(dense_68/bias/Regularizer/Const:output:0!dense_68/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_68/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp5dense_68_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_68/bias/Regularizer/L2LossL2Loss7dense_68/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_68/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_68/bias/Regularizer/mul_1Mul*dense_68/bias/Regularizer/mul_1/x:output:0)dense_68/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/add_1AddV2!dense_68/bias/Regularizer/add:z:0#dense_68/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_68/bias/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp-^dense_68/bias/Regularizer/Abs/ReadVariableOp0^dense_68/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,dense_68/bias/Regularizer/Abs/ReadVariableOp,dense_68/bias/Regularizer/Abs/ReadVariableOp2b
/dense_68/bias/Regularizer/L2Loss/ReadVariableOp/dense_68/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
__inference_loss_fn_10_21799151K
7dense_70_kernel_regularizer_abs_readvariableop_resource:
��
identity��.dense_70/kernel/Regularizer/Abs/ReadVariableOp�1dense_70/kernel/Regularizer/L2Loss/ReadVariableOpf
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_70/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_70_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_70/kernel/Regularizer/AbsAbs6dense_70/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_70/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_70/kernel/Regularizer/SumSum#dense_70/kernel/Regularizer/Abs:y:0,dense_70/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/kernel/Regularizer/addAddV2*dense_70/kernel/Regularizer/Const:output:0#dense_70/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_70_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_70/kernel/Regularizer/L2LossL2Loss9dense_70/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_70/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_70/kernel/Regularizer/mul_1Mul,dense_70/kernel/Regularizer/mul_1/x:output:0+dense_70/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_70/kernel/Regularizer/add_1AddV2#dense_70/kernel/Regularizer/add:z:0%dense_70/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_70/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^dense_70/kernel/Regularizer/Abs/ReadVariableOp2^dense_70/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_70/kernel/Regularizer/Abs/ReadVariableOp.dense_70/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�,
�
F__inference_dense_67_layer_call_and_return_conditional_losses_21797676

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_67/bias/Regularizer/Abs/ReadVariableOp�/dense_67/bias/Regularizer/L2Loss/ReadVariableOp�.dense_67/kernel/Regularizer/Abs/ReadVariableOp�1dense_67/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_67/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_67/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_67/kernel/Regularizer/AbsAbs6dense_67/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_67/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_67/kernel/Regularizer/SumSum#dense_67/kernel/Regularizer/Abs:y:0,dense_67/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_67/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/kernel/Regularizer/mulMul*dense_67/kernel/Regularizer/mul/x:output:0(dense_67/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/kernel/Regularizer/addAddV2*dense_67/kernel/Regularizer/Const:output:0#dense_67/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_67/kernel/Regularizer/L2LossL2Loss9dense_67/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_67/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_67/kernel/Regularizer/mul_1Mul,dense_67/kernel/Regularizer/mul_1/x:output:0+dense_67/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_67/kernel/Regularizer/add_1AddV2#dense_67/kernel/Regularizer/add:z:0%dense_67/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_67/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_67/bias/Regularizer/AbsAbs4dense_67/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_67/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_67/bias/Regularizer/SumSum!dense_67/bias/Regularizer/Abs:y:0*dense_67/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/bias/Regularizer/mulMul(dense_67/bias/Regularizer/mul/x:output:0&dense_67/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/addAddV2(dense_67/bias/Regularizer/Const:output:0!dense_67/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_67/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_67/bias/Regularizer/L2LossL2Loss7dense_67/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_67/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_67/bias/Regularizer/mul_1Mul*dense_67/bias/Regularizer/mul_1/x:output:0)dense_67/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/add_1AddV2!dense_67/bias/Regularizer/add:z:0#dense_67/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_67/bias/Regularizer/Abs/ReadVariableOp0^dense_67/bias/Regularizer/L2Loss/ReadVariableOp/^dense_67/kernel/Regularizer/Abs/ReadVariableOp2^dense_67/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_67/bias/Regularizer/Abs/ReadVariableOp,dense_67/bias/Regularizer/Abs/ReadVariableOp2b
/dense_67/bias/Regularizer/L2Loss/ReadVariableOp/dense_67/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_67/kernel/Regularizer/Abs/ReadVariableOp.dense_67/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_21798981J
7dense_65_kernel_regularizer_abs_readvariableop_resource:	�
identity��.dense_65/kernel/Regularizer/Abs/ReadVariableOp�1dense_65/kernel/Regularizer/L2Loss/ReadVariableOpf
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_65/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_65_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_65/kernel/Regularizer/AbsAbs6dense_65/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�t
#dense_65/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_65/kernel/Regularizer/SumSum#dense_65/kernel/Regularizer/Abs:y:0,dense_65/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/kernel/Regularizer/addAddV2*dense_65/kernel/Regularizer/Const:output:0#dense_65/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_65_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_65/kernel/Regularizer/L2LossL2Loss9dense_65/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_65/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_65/kernel/Regularizer/mul_1Mul,dense_65/kernel/Regularizer/mul_1/x:output:0+dense_65/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_65/kernel/Regularizer/add_1AddV2#dense_65/kernel/Regularizer/add:z:0%dense_65/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_65/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^dense_65/kernel/Regularizer/Abs/ReadVariableOp2^dense_65/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_65/kernel/Regularizer/Abs/ReadVariableOp.dense_65/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�(
�
__inference_adapt_step_1516630
iterator%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:���������*&
output_shapes
:���������*
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 o
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	:��Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22$
AssignVariableOpAssignVariableOp2"
IteratorGetNextIteratorGetNext2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
iterator
�
�
+__inference_dense_67_layer_call_fn_21798770

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_67_layer_call_and_return_conditional_losses_21797676p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798766:($
"
_user_specified_name
21798764:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_sequential_13_layer_call_fn_21798256
normalization_input
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_21798182o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798252:($
"
_user_specified_name
21798250:($
"
_user_specified_name
21798248:($
"
_user_specified_name
21798246:($
"
_user_specified_name
21798244:($
"
_user_specified_name
21798242:(
$
"
_user_specified_name
21798240:(	$
"
_user_specified_name
21798238:($
"
_user_specified_name
21798236:($
"
_user_specified_name
21798234:($
"
_user_specified_name
21798232:($
"
_user_specified_name
21798230:($
"
_user_specified_name
21798228:($
"
_user_specified_name
21798226:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
__inference_loss_fn_1_21798998D
5dense_65_bias_regularizer_abs_readvariableop_resource:	�
identity��,dense_65/bias/Regularizer/Abs/ReadVariableOp�/dense_65/bias/Regularizer/L2Loss/ReadVariableOpd
dense_65/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_65/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_65_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_65/bias/Regularizer/AbsAbs4dense_65/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_65/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_65/bias/Regularizer/SumSum!dense_65/bias/Regularizer/Abs:y:0*dense_65/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/bias/Regularizer/mulMul(dense_65/bias/Regularizer/mul/x:output:0&dense_65/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/addAddV2(dense_65/bias/Regularizer/Const:output:0!dense_65/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_65/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp5dense_65_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_65/bias/Regularizer/L2LossL2Loss7dense_65/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_65/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_65/bias/Regularizer/mul_1Mul*dense_65/bias/Regularizer/mul_1/x:output:0)dense_65/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/add_1AddV2!dense_65/bias/Regularizer/add:z:0#dense_65/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_65/bias/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp-^dense_65/bias/Regularizer/Abs/ReadVariableOp0^dense_65/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,dense_65/bias/Regularizer/Abs/ReadVariableOp,dense_65/bias/Regularizer/Abs/ReadVariableOp2b
/dense_65/bias/Regularizer/L2Loss/ReadVariableOp/dense_65/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
��
�
K__inference_sequential_13_layer_call_and_return_conditional_losses_21798182
normalization_input
normalization_sub_y
normalization_sqrt_x$
dense_65_21797990:	� 
dense_65_21797992:	�%
dense_66_21797995:
�� 
dense_66_21797997:	�%
dense_67_21798000:
�� 
dense_67_21798002:	�%
dense_68_21798005:
�� 
dense_68_21798007:	�%
dense_69_21798010:
�� 
dense_69_21798012:	�%
dense_70_21798015:
�� 
dense_70_21798017:	�$
dense_71_21798020:	�
dense_71_21798022:
identity�� dense_65/StatefulPartitionedCall�,dense_65/bias/Regularizer/Abs/ReadVariableOp�/dense_65/bias/Regularizer/L2Loss/ReadVariableOp�.dense_65/kernel/Regularizer/Abs/ReadVariableOp�1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp� dense_66/StatefulPartitionedCall�,dense_66/bias/Regularizer/Abs/ReadVariableOp�/dense_66/bias/Regularizer/L2Loss/ReadVariableOp�.dense_66/kernel/Regularizer/Abs/ReadVariableOp�1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp� dense_67/StatefulPartitionedCall�,dense_67/bias/Regularizer/Abs/ReadVariableOp�/dense_67/bias/Regularizer/L2Loss/ReadVariableOp�.dense_67/kernel/Regularizer/Abs/ReadVariableOp�1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp� dense_68/StatefulPartitionedCall�,dense_68/bias/Regularizer/Abs/ReadVariableOp�/dense_68/bias/Regularizer/L2Loss/ReadVariableOp�.dense_68/kernel/Regularizer/Abs/ReadVariableOp�1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp� dense_69/StatefulPartitionedCall�,dense_69/bias/Regularizer/Abs/ReadVariableOp�/dense_69/bias/Regularizer/L2Loss/ReadVariableOp�.dense_69/kernel/Regularizer/Abs/ReadVariableOp�1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp� dense_70/StatefulPartitionedCall�,dense_70/bias/Regularizer/Abs/ReadVariableOp�/dense_70/bias/Regularizer/L2Loss/ReadVariableOp�.dense_70/kernel/Regularizer/Abs/ReadVariableOp�1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp� dense_71/StatefulPartitionedCallt
normalization/subSubnormalization_inputnormalization_sub_y*
T0*'
_output_shapes
:���������Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
 dense_65/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_65_21797990dense_65_21797992*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_21797592�
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_21797995dense_66_21797997*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_21797634�
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_21798000dense_67_21798002*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_67_layer_call_and_return_conditional_losses_21797676�
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_21798005dense_68_21798007*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_68_layer_call_and_return_conditional_losses_21797718�
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_21798010dense_69_21798012*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_69_layer_call_and_return_conditional_losses_21797760�
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_21798015dense_70_21798017*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_70_layer_call_and_return_conditional_losses_21797802�
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_21798020dense_71_21798022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_71_layer_call_and_return_conditional_losses_21797817f
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_65/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_65_21797990*
_output_shapes
:	�*
dtype0�
dense_65/kernel/Regularizer/AbsAbs6dense_65/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�t
#dense_65/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_65/kernel/Regularizer/SumSum#dense_65/kernel/Regularizer/Abs:y:0,dense_65/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/kernel/Regularizer/addAddV2*dense_65/kernel/Regularizer/Const:output:0#dense_65/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_65_21797990*
_output_shapes
:	�*
dtype0�
"dense_65/kernel/Regularizer/L2LossL2Loss9dense_65/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_65/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_65/kernel/Regularizer/mul_1Mul,dense_65/kernel/Regularizer/mul_1/x:output:0+dense_65/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_65/kernel/Regularizer/add_1AddV2#dense_65/kernel/Regularizer/add:z:0%dense_65/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_65/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_65_21797992*
_output_shapes	
:�*
dtype0�
dense_65/bias/Regularizer/AbsAbs4dense_65/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_65/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_65/bias/Regularizer/SumSum!dense_65/bias/Regularizer/Abs:y:0*dense_65/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/bias/Regularizer/mulMul(dense_65/bias/Regularizer/mul/x:output:0&dense_65/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/addAddV2(dense_65/bias/Regularizer/Const:output:0!dense_65/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_65/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_65_21797992*
_output_shapes	
:�*
dtype0�
 dense_65/bias/Regularizer/L2LossL2Loss7dense_65/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_65/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_65/bias/Regularizer/mul_1Mul*dense_65/bias/Regularizer/mul_1/x:output:0)dense_65/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/add_1AddV2!dense_65/bias/Regularizer/add:z:0#dense_65/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_66/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_66/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_66_21797995* 
_output_shapes
:
��*
dtype0�
dense_66/kernel/Regularizer/AbsAbs6dense_66/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_66/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_66/kernel/Regularizer/SumSum#dense_66/kernel/Regularizer/Abs:y:0,dense_66/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_66/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/kernel/Regularizer/mulMul*dense_66/kernel/Regularizer/mul/x:output:0(dense_66/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/kernel/Regularizer/addAddV2*dense_66/kernel/Regularizer/Const:output:0#dense_66/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_66_21797995* 
_output_shapes
:
��*
dtype0�
"dense_66/kernel/Regularizer/L2LossL2Loss9dense_66/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_66/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_66/kernel/Regularizer/mul_1Mul,dense_66/kernel/Regularizer/mul_1/x:output:0+dense_66/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_66/kernel/Regularizer/add_1AddV2#dense_66/kernel/Regularizer/add:z:0%dense_66/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_66/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_66_21797997*
_output_shapes	
:�*
dtype0�
dense_66/bias/Regularizer/AbsAbs4dense_66/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_66/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_66/bias/Regularizer/SumSum!dense_66/bias/Regularizer/Abs:y:0*dense_66/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/bias/Regularizer/mulMul(dense_66/bias/Regularizer/mul/x:output:0&dense_66/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/addAddV2(dense_66/bias/Regularizer/Const:output:0!dense_66/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_66/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_66_21797997*
_output_shapes	
:�*
dtype0�
 dense_66/bias/Regularizer/L2LossL2Loss7dense_66/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_66/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_66/bias/Regularizer/mul_1Mul*dense_66/bias/Regularizer/mul_1/x:output:0)dense_66/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/add_1AddV2!dense_66/bias/Regularizer/add:z:0#dense_66/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_67/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_67/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_67_21798000* 
_output_shapes
:
��*
dtype0�
dense_67/kernel/Regularizer/AbsAbs6dense_67/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_67/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_67/kernel/Regularizer/SumSum#dense_67/kernel/Regularizer/Abs:y:0,dense_67/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_67/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/kernel/Regularizer/mulMul*dense_67/kernel/Regularizer/mul/x:output:0(dense_67/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/kernel/Regularizer/addAddV2*dense_67/kernel/Regularizer/Const:output:0#dense_67/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_67_21798000* 
_output_shapes
:
��*
dtype0�
"dense_67/kernel/Regularizer/L2LossL2Loss9dense_67/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_67/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_67/kernel/Regularizer/mul_1Mul,dense_67/kernel/Regularizer/mul_1/x:output:0+dense_67/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_67/kernel/Regularizer/add_1AddV2#dense_67/kernel/Regularizer/add:z:0%dense_67/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_67/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_67_21798002*
_output_shapes	
:�*
dtype0�
dense_67/bias/Regularizer/AbsAbs4dense_67/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_67/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_67/bias/Regularizer/SumSum!dense_67/bias/Regularizer/Abs:y:0*dense_67/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/bias/Regularizer/mulMul(dense_67/bias/Regularizer/mul/x:output:0&dense_67/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/addAddV2(dense_67/bias/Regularizer/Const:output:0!dense_67/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_67/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_67_21798002*
_output_shapes	
:�*
dtype0�
 dense_67/bias/Regularizer/L2LossL2Loss7dense_67/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_67/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_67/bias/Regularizer/mul_1Mul*dense_67/bias/Regularizer/mul_1/x:output:0)dense_67/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/add_1AddV2!dense_67/bias/Regularizer/add:z:0#dense_67/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_68/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_68/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_68_21798005* 
_output_shapes
:
��*
dtype0�
dense_68/kernel/Regularizer/AbsAbs6dense_68/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_68/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_68/kernel/Regularizer/SumSum#dense_68/kernel/Regularizer/Abs:y:0,dense_68/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_68/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/kernel/Regularizer/mulMul*dense_68/kernel/Regularizer/mul/x:output:0(dense_68/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/kernel/Regularizer/addAddV2*dense_68/kernel/Regularizer/Const:output:0#dense_68/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_68_21798005* 
_output_shapes
:
��*
dtype0�
"dense_68/kernel/Regularizer/L2LossL2Loss9dense_68/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_68/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_68/kernel/Regularizer/mul_1Mul,dense_68/kernel/Regularizer/mul_1/x:output:0+dense_68/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_68/kernel/Regularizer/add_1AddV2#dense_68/kernel/Regularizer/add:z:0%dense_68/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_68/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_68_21798007*
_output_shapes	
:�*
dtype0�
dense_68/bias/Regularizer/AbsAbs4dense_68/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_68/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_68/bias/Regularizer/SumSum!dense_68/bias/Regularizer/Abs:y:0*dense_68/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/bias/Regularizer/mulMul(dense_68/bias/Regularizer/mul/x:output:0&dense_68/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/addAddV2(dense_68/bias/Regularizer/Const:output:0!dense_68/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_68/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_68_21798007*
_output_shapes	
:�*
dtype0�
 dense_68/bias/Regularizer/L2LossL2Loss7dense_68/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_68/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_68/bias/Regularizer/mul_1Mul*dense_68/bias/Regularizer/mul_1/x:output:0)dense_68/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/add_1AddV2!dense_68/bias/Regularizer/add:z:0#dense_68/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_69/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_69/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_69_21798010* 
_output_shapes
:
��*
dtype0�
dense_69/kernel/Regularizer/AbsAbs6dense_69/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_69/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_69/kernel/Regularizer/SumSum#dense_69/kernel/Regularizer/Abs:y:0,dense_69/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_69/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/kernel/Regularizer/mulMul*dense_69/kernel/Regularizer/mul/x:output:0(dense_69/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/kernel/Regularizer/addAddV2*dense_69/kernel/Regularizer/Const:output:0#dense_69/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_69_21798010* 
_output_shapes
:
��*
dtype0�
"dense_69/kernel/Regularizer/L2LossL2Loss9dense_69/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_69/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_69/kernel/Regularizer/mul_1Mul,dense_69/kernel/Regularizer/mul_1/x:output:0+dense_69/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_69/kernel/Regularizer/add_1AddV2#dense_69/kernel/Regularizer/add:z:0%dense_69/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_69/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_69_21798012*
_output_shapes	
:�*
dtype0�
dense_69/bias/Regularizer/AbsAbs4dense_69/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_69/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_69/bias/Regularizer/SumSum!dense_69/bias/Regularizer/Abs:y:0*dense_69/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/bias/Regularizer/mulMul(dense_69/bias/Regularizer/mul/x:output:0&dense_69/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/addAddV2(dense_69/bias/Regularizer/Const:output:0!dense_69/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_69/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_69_21798012*
_output_shapes	
:�*
dtype0�
 dense_69/bias/Regularizer/L2LossL2Loss7dense_69/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_69/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_69/bias/Regularizer/mul_1Mul*dense_69/bias/Regularizer/mul_1/x:output:0)dense_69/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/add_1AddV2!dense_69/bias/Regularizer/add:z:0#dense_69/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_70/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_70_21798015* 
_output_shapes
:
��*
dtype0�
dense_70/kernel/Regularizer/AbsAbs6dense_70/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_70/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_70/kernel/Regularizer/SumSum#dense_70/kernel/Regularizer/Abs:y:0,dense_70/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/kernel/Regularizer/addAddV2*dense_70/kernel/Regularizer/Const:output:0#dense_70/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_70_21798015* 
_output_shapes
:
��*
dtype0�
"dense_70/kernel/Regularizer/L2LossL2Loss9dense_70/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_70/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_70/kernel/Regularizer/mul_1Mul,dense_70/kernel/Regularizer/mul_1/x:output:0+dense_70/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_70/kernel/Regularizer/add_1AddV2#dense_70/kernel/Regularizer/add:z:0%dense_70/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_70/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_70_21798017*
_output_shapes	
:�*
dtype0�
dense_70/bias/Regularizer/AbsAbs4dense_70/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_70/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_70/bias/Regularizer/SumSum!dense_70/bias/Regularizer/Abs:y:0*dense_70/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/bias/Regularizer/mulMul(dense_70/bias/Regularizer/mul/x:output:0&dense_70/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/addAddV2(dense_70/bias/Regularizer/Const:output:0!dense_70/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_70/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_70_21798017*
_output_shapes	
:�*
dtype0�
 dense_70/bias/Regularizer/L2LossL2Loss7dense_70/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_70/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_70/bias/Regularizer/mul_1Mul*dense_70/bias/Regularizer/mul_1/x:output:0)dense_70/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/add_1AddV2!dense_70/bias/Regularizer/add:z:0#dense_70/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_65/StatefulPartitionedCall-^dense_65/bias/Regularizer/Abs/ReadVariableOp0^dense_65/bias/Regularizer/L2Loss/ReadVariableOp/^dense_65/kernel/Regularizer/Abs/ReadVariableOp2^dense_65/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_66/StatefulPartitionedCall-^dense_66/bias/Regularizer/Abs/ReadVariableOp0^dense_66/bias/Regularizer/L2Loss/ReadVariableOp/^dense_66/kernel/Regularizer/Abs/ReadVariableOp2^dense_66/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_67/StatefulPartitionedCall-^dense_67/bias/Regularizer/Abs/ReadVariableOp0^dense_67/bias/Regularizer/L2Loss/ReadVariableOp/^dense_67/kernel/Regularizer/Abs/ReadVariableOp2^dense_67/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_68/StatefulPartitionedCall-^dense_68/bias/Regularizer/Abs/ReadVariableOp0^dense_68/bias/Regularizer/L2Loss/ReadVariableOp/^dense_68/kernel/Regularizer/Abs/ReadVariableOp2^dense_68/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_69/StatefulPartitionedCall-^dense_69/bias/Regularizer/Abs/ReadVariableOp0^dense_69/bias/Regularizer/L2Loss/ReadVariableOp/^dense_69/kernel/Regularizer/Abs/ReadVariableOp2^dense_69/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_70/StatefulPartitionedCall-^dense_70/bias/Regularizer/Abs/ReadVariableOp0^dense_70/bias/Regularizer/L2Loss/ReadVariableOp/^dense_70/kernel/Regularizer/Abs/ReadVariableOp2^dense_70/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_71/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2\
,dense_65/bias/Regularizer/Abs/ReadVariableOp,dense_65/bias/Regularizer/Abs/ReadVariableOp2b
/dense_65/bias/Regularizer/L2Loss/ReadVariableOp/dense_65/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_65/kernel/Regularizer/Abs/ReadVariableOp.dense_65/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2\
,dense_66/bias/Regularizer/Abs/ReadVariableOp,dense_66/bias/Regularizer/Abs/ReadVariableOp2b
/dense_66/bias/Regularizer/L2Loss/ReadVariableOp/dense_66/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_66/kernel/Regularizer/Abs/ReadVariableOp.dense_66/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2\
,dense_67/bias/Regularizer/Abs/ReadVariableOp,dense_67/bias/Regularizer/Abs/ReadVariableOp2b
/dense_67/bias/Regularizer/L2Loss/ReadVariableOp/dense_67/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_67/kernel/Regularizer/Abs/ReadVariableOp.dense_67/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2\
,dense_68/bias/Regularizer/Abs/ReadVariableOp,dense_68/bias/Regularizer/Abs/ReadVariableOp2b
/dense_68/bias/Regularizer/L2Loss/ReadVariableOp/dense_68/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_68/kernel/Regularizer/Abs/ReadVariableOp.dense_68/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2\
,dense_69/bias/Regularizer/Abs/ReadVariableOp,dense_69/bias/Regularizer/Abs/ReadVariableOp2b
/dense_69/bias/Regularizer/L2Loss/ReadVariableOp/dense_69/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_69/kernel/Regularizer/Abs/ReadVariableOp.dense_69/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2\
,dense_70/bias/Regularizer/Abs/ReadVariableOp,dense_70/bias/Regularizer/Abs/ReadVariableOp2b
/dense_70/bias/Regularizer/L2Loss/ReadVariableOp/dense_70/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_70/kernel/Regularizer/Abs/ReadVariableOp.dense_70/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall:($
"
_user_specified_name
21798022:($
"
_user_specified_name
21798020:($
"
_user_specified_name
21798017:($
"
_user_specified_name
21798015:($
"
_user_specified_name
21798012:($
"
_user_specified_name
21798010:(
$
"
_user_specified_name
21798007:(	$
"
_user_specified_name
21798005:($
"
_user_specified_name
21798002:($
"
_user_specified_name
21798000:($
"
_user_specified_name
21797997:($
"
_user_specified_name
21797995:($
"
_user_specified_name
21797992:($
"
_user_specified_name
21797990:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
+__inference_dense_71_layer_call_fn_21798954

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_71_layer_call_and_return_conditional_losses_21797817o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798950:($
"
_user_specified_name
21798948:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
F__inference_dense_68_layer_call_and_return_conditional_losses_21798853

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_68/bias/Regularizer/Abs/ReadVariableOp�/dense_68/bias/Regularizer/L2Loss/ReadVariableOp�.dense_68/kernel/Regularizer/Abs/ReadVariableOp�1dense_68/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_68/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_68/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_68/kernel/Regularizer/AbsAbs6dense_68/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_68/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_68/kernel/Regularizer/SumSum#dense_68/kernel/Regularizer/Abs:y:0,dense_68/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_68/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/kernel/Regularizer/mulMul*dense_68/kernel/Regularizer/mul/x:output:0(dense_68/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/kernel/Regularizer/addAddV2*dense_68/kernel/Regularizer/Const:output:0#dense_68/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_68/kernel/Regularizer/L2LossL2Loss9dense_68/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_68/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_68/kernel/Regularizer/mul_1Mul,dense_68/kernel/Regularizer/mul_1/x:output:0+dense_68/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_68/kernel/Regularizer/add_1AddV2#dense_68/kernel/Regularizer/add:z:0%dense_68/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_68/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_68/bias/Regularizer/AbsAbs4dense_68/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_68/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_68/bias/Regularizer/SumSum!dense_68/bias/Regularizer/Abs:y:0*dense_68/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/bias/Regularizer/mulMul(dense_68/bias/Regularizer/mul/x:output:0&dense_68/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/addAddV2(dense_68/bias/Regularizer/Const:output:0!dense_68/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_68/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_68/bias/Regularizer/L2LossL2Loss7dense_68/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_68/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_68/bias/Regularizer/mul_1Mul*dense_68/bias/Regularizer/mul_1/x:output:0)dense_68/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/add_1AddV2!dense_68/bias/Regularizer/add:z:0#dense_68/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_68/bias/Regularizer/Abs/ReadVariableOp0^dense_68/bias/Regularizer/L2Loss/ReadVariableOp/^dense_68/kernel/Regularizer/Abs/ReadVariableOp2^dense_68/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_68/bias/Regularizer/Abs/ReadVariableOp,dense_68/bias/Regularizer/Abs/ReadVariableOp2b
/dense_68/bias/Regularizer/L2Loss/ReadVariableOp/dense_68/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_68/kernel/Regularizer/Abs/ReadVariableOp.dense_68/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
F__inference_dense_69_layer_call_and_return_conditional_losses_21798899

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_69/bias/Regularizer/Abs/ReadVariableOp�/dense_69/bias/Regularizer/L2Loss/ReadVariableOp�.dense_69/kernel/Regularizer/Abs/ReadVariableOp�1dense_69/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_69/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_69/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_69/kernel/Regularizer/AbsAbs6dense_69/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_69/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_69/kernel/Regularizer/SumSum#dense_69/kernel/Regularizer/Abs:y:0,dense_69/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_69/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/kernel/Regularizer/mulMul*dense_69/kernel/Regularizer/mul/x:output:0(dense_69/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/kernel/Regularizer/addAddV2*dense_69/kernel/Regularizer/Const:output:0#dense_69/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_69/kernel/Regularizer/L2LossL2Loss9dense_69/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_69/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_69/kernel/Regularizer/mul_1Mul,dense_69/kernel/Regularizer/mul_1/x:output:0+dense_69/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_69/kernel/Regularizer/add_1AddV2#dense_69/kernel/Regularizer/add:z:0%dense_69/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_69/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_69/bias/Regularizer/AbsAbs4dense_69/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_69/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_69/bias/Regularizer/SumSum!dense_69/bias/Regularizer/Abs:y:0*dense_69/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/bias/Regularizer/mulMul(dense_69/bias/Regularizer/mul/x:output:0&dense_69/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/addAddV2(dense_69/bias/Regularizer/Const:output:0!dense_69/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_69/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_69/bias/Regularizer/L2LossL2Loss7dense_69/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_69/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_69/bias/Regularizer/mul_1Mul*dense_69/bias/Regularizer/mul_1/x:output:0)dense_69/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/add_1AddV2!dense_69/bias/Regularizer/add:z:0#dense_69/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_69/bias/Regularizer/Abs/ReadVariableOp0^dense_69/bias/Regularizer/L2Loss/ReadVariableOp/^dense_69/kernel/Regularizer/Abs/ReadVariableOp2^dense_69/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_69/bias/Regularizer/Abs/ReadVariableOp,dense_69/bias/Regularizer/Abs/ReadVariableOp2b
/dense_69/bias/Regularizer/L2Loss/ReadVariableOp/dense_69/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_69/kernel/Regularizer/Abs/ReadVariableOp.dense_69/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
F__inference_dense_70_layer_call_and_return_conditional_losses_21797802

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_70/bias/Regularizer/Abs/ReadVariableOp�/dense_70/bias/Regularizer/L2Loss/ReadVariableOp�.dense_70/kernel/Regularizer/Abs/ReadVariableOp�1dense_70/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_70/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_70/kernel/Regularizer/AbsAbs6dense_70/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_70/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_70/kernel/Regularizer/SumSum#dense_70/kernel/Regularizer/Abs:y:0,dense_70/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/kernel/Regularizer/addAddV2*dense_70/kernel/Regularizer/Const:output:0#dense_70/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_70/kernel/Regularizer/L2LossL2Loss9dense_70/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_70/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_70/kernel/Regularizer/mul_1Mul,dense_70/kernel/Regularizer/mul_1/x:output:0+dense_70/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_70/kernel/Regularizer/add_1AddV2#dense_70/kernel/Regularizer/add:z:0%dense_70/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_70/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_70/bias/Regularizer/AbsAbs4dense_70/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_70/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_70/bias/Regularizer/SumSum!dense_70/bias/Regularizer/Abs:y:0*dense_70/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/bias/Regularizer/mulMul(dense_70/bias/Regularizer/mul/x:output:0&dense_70/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/addAddV2(dense_70/bias/Regularizer/Const:output:0!dense_70/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_70/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_70/bias/Regularizer/L2LossL2Loss7dense_70/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_70/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_70/bias/Regularizer/mul_1Mul*dense_70/bias/Regularizer/mul_1/x:output:0)dense_70/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/add_1AddV2!dense_70/bias/Regularizer/add:z:0#dense_70/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_70/bias/Regularizer/Abs/ReadVariableOp0^dense_70/bias/Regularizer/L2Loss/ReadVariableOp/^dense_70/kernel/Regularizer/Abs/ReadVariableOp2^dense_70/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_70/bias/Regularizer/Abs/ReadVariableOp,dense_70/bias/Regularizer/Abs/ReadVariableOp2b
/dense_70/bias/Regularizer/L2Loss/ReadVariableOp/dense_70/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_70/kernel/Regularizer/Abs/ReadVariableOp.dense_70/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
F__inference_dense_65_layer_call_and_return_conditional_losses_21798715

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_65/bias/Regularizer/Abs/ReadVariableOp�/dense_65/bias/Regularizer/L2Loss/ReadVariableOp�.dense_65/kernel/Regularizer/Abs/ReadVariableOp�1dense_65/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_65/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_65/kernel/Regularizer/AbsAbs6dense_65/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�t
#dense_65/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_65/kernel/Regularizer/SumSum#dense_65/kernel/Regularizer/Abs:y:0,dense_65/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/kernel/Regularizer/addAddV2*dense_65/kernel/Regularizer/Const:output:0#dense_65/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_65/kernel/Regularizer/L2LossL2Loss9dense_65/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_65/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_65/kernel/Regularizer/mul_1Mul,dense_65/kernel/Regularizer/mul_1/x:output:0+dense_65/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_65/kernel/Regularizer/add_1AddV2#dense_65/kernel/Regularizer/add:z:0%dense_65/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_65/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_65/bias/Regularizer/AbsAbs4dense_65/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_65/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_65/bias/Regularizer/SumSum!dense_65/bias/Regularizer/Abs:y:0*dense_65/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/bias/Regularizer/mulMul(dense_65/bias/Regularizer/mul/x:output:0&dense_65/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/addAddV2(dense_65/bias/Regularizer/Const:output:0!dense_65/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_65/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_65/bias/Regularizer/L2LossL2Loss7dense_65/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_65/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_65/bias/Regularizer/mul_1Mul*dense_65/bias/Regularizer/mul_1/x:output:0)dense_65/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/add_1AddV2!dense_65/bias/Regularizer/add:z:0#dense_65/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_65/bias/Regularizer/Abs/ReadVariableOp0^dense_65/bias/Regularizer/L2Loss/ReadVariableOp/^dense_65/kernel/Regularizer/Abs/ReadVariableOp2^dense_65/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_65/bias/Regularizer/Abs/ReadVariableOp,dense_65/bias/Regularizer/Abs/ReadVariableOp2b
/dense_65/bias/Regularizer/L2Loss/ReadVariableOp/dense_65/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_65/kernel/Regularizer/Abs/ReadVariableOp.dense_65/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_21799015K
7dense_66_kernel_regularizer_abs_readvariableop_resource:
��
identity��.dense_66/kernel/Regularizer/Abs/ReadVariableOp�1dense_66/kernel/Regularizer/L2Loss/ReadVariableOpf
!dense_66/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_66/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_66_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_66/kernel/Regularizer/AbsAbs6dense_66/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_66/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_66/kernel/Regularizer/SumSum#dense_66/kernel/Regularizer/Abs:y:0,dense_66/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_66/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/kernel/Regularizer/mulMul*dense_66/kernel/Regularizer/mul/x:output:0(dense_66/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/kernel/Regularizer/addAddV2*dense_66/kernel/Regularizer/Const:output:0#dense_66/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_66_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_66/kernel/Regularizer/L2LossL2Loss9dense_66/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_66/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_66/kernel/Regularizer/mul_1Mul,dense_66/kernel/Regularizer/mul_1/x:output:0+dense_66/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_66/kernel/Regularizer/add_1AddV2#dense_66/kernel/Regularizer/add:z:0%dense_66/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_66/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^dense_66/kernel/Regularizer/Abs/ReadVariableOp2^dense_66/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_66/kernel/Regularizer/Abs/ReadVariableOp.dense_66/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�,
�
F__inference_dense_65_layer_call_and_return_conditional_losses_21797592

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_65/bias/Regularizer/Abs/ReadVariableOp�/dense_65/bias/Regularizer/L2Loss/ReadVariableOp�.dense_65/kernel/Regularizer/Abs/ReadVariableOp�1dense_65/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_65/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_65/kernel/Regularizer/AbsAbs6dense_65/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�t
#dense_65/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_65/kernel/Regularizer/SumSum#dense_65/kernel/Regularizer/Abs:y:0,dense_65/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/kernel/Regularizer/addAddV2*dense_65/kernel/Regularizer/Const:output:0#dense_65/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
"dense_65/kernel/Regularizer/L2LossL2Loss9dense_65/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_65/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_65/kernel/Regularizer/mul_1Mul,dense_65/kernel/Regularizer/mul_1/x:output:0+dense_65/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_65/kernel/Regularizer/add_1AddV2#dense_65/kernel/Regularizer/add:z:0%dense_65/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_65/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_65/bias/Regularizer/AbsAbs4dense_65/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_65/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_65/bias/Regularizer/SumSum!dense_65/bias/Regularizer/Abs:y:0*dense_65/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/bias/Regularizer/mulMul(dense_65/bias/Regularizer/mul/x:output:0&dense_65/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/addAddV2(dense_65/bias/Regularizer/Const:output:0!dense_65/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_65/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_65/bias/Regularizer/L2LossL2Loss7dense_65/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_65/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_65/bias/Regularizer/mul_1Mul*dense_65/bias/Regularizer/mul_1/x:output:0)dense_65/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/add_1AddV2!dense_65/bias/Regularizer/add:z:0#dense_65/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_65/bias/Regularizer/Abs/ReadVariableOp0^dense_65/bias/Regularizer/L2Loss/ReadVariableOp/^dense_65/kernel/Regularizer/Abs/ReadVariableOp2^dense_65/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_65/bias/Regularizer/Abs/ReadVariableOp,dense_65/bias/Regularizer/Abs/ReadVariableOp2b
/dense_65/bias/Regularizer/L2Loss/ReadVariableOp/dense_65/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_65/kernel/Regularizer/Abs/ReadVariableOp.dense_65/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
K__inference_sequential_13_layer_call_and_return_conditional_losses_21797980
normalization_input
normalization_sub_y
normalization_sqrt_x$
dense_65_21797593:	� 
dense_65_21797595:	�%
dense_66_21797635:
�� 
dense_66_21797637:	�%
dense_67_21797677:
�� 
dense_67_21797679:	�%
dense_68_21797719:
�� 
dense_68_21797721:	�%
dense_69_21797761:
�� 
dense_69_21797763:	�%
dense_70_21797803:
�� 
dense_70_21797805:	�$
dense_71_21797818:	�
dense_71_21797820:
identity�� dense_65/StatefulPartitionedCall�,dense_65/bias/Regularizer/Abs/ReadVariableOp�/dense_65/bias/Regularizer/L2Loss/ReadVariableOp�.dense_65/kernel/Regularizer/Abs/ReadVariableOp�1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp� dense_66/StatefulPartitionedCall�,dense_66/bias/Regularizer/Abs/ReadVariableOp�/dense_66/bias/Regularizer/L2Loss/ReadVariableOp�.dense_66/kernel/Regularizer/Abs/ReadVariableOp�1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp� dense_67/StatefulPartitionedCall�,dense_67/bias/Regularizer/Abs/ReadVariableOp�/dense_67/bias/Regularizer/L2Loss/ReadVariableOp�.dense_67/kernel/Regularizer/Abs/ReadVariableOp�1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp� dense_68/StatefulPartitionedCall�,dense_68/bias/Regularizer/Abs/ReadVariableOp�/dense_68/bias/Regularizer/L2Loss/ReadVariableOp�.dense_68/kernel/Regularizer/Abs/ReadVariableOp�1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp� dense_69/StatefulPartitionedCall�,dense_69/bias/Regularizer/Abs/ReadVariableOp�/dense_69/bias/Regularizer/L2Loss/ReadVariableOp�.dense_69/kernel/Regularizer/Abs/ReadVariableOp�1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp� dense_70/StatefulPartitionedCall�,dense_70/bias/Regularizer/Abs/ReadVariableOp�/dense_70/bias/Regularizer/L2Loss/ReadVariableOp�.dense_70/kernel/Regularizer/Abs/ReadVariableOp�1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp� dense_71/StatefulPartitionedCallt
normalization/subSubnormalization_inputnormalization_sub_y*
T0*'
_output_shapes
:���������Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
 dense_65/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_65_21797593dense_65_21797595*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_21797592�
 dense_66/StatefulPartitionedCallStatefulPartitionedCall)dense_65/StatefulPartitionedCall:output:0dense_66_21797635dense_66_21797637*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_21797634�
 dense_67/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0dense_67_21797677dense_67_21797679*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_67_layer_call_and_return_conditional_losses_21797676�
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_21797719dense_68_21797721*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_68_layer_call_and_return_conditional_losses_21797718�
 dense_69/StatefulPartitionedCallStatefulPartitionedCall)dense_68/StatefulPartitionedCall:output:0dense_69_21797761dense_69_21797763*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_69_layer_call_and_return_conditional_losses_21797760�
 dense_70/StatefulPartitionedCallStatefulPartitionedCall)dense_69/StatefulPartitionedCall:output:0dense_70_21797803dense_70_21797805*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_70_layer_call_and_return_conditional_losses_21797802�
 dense_71/StatefulPartitionedCallStatefulPartitionedCall)dense_70/StatefulPartitionedCall:output:0dense_71_21797818dense_71_21797820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_71_layer_call_and_return_conditional_losses_21797817f
!dense_65/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_65/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_65_21797593*
_output_shapes
:	�*
dtype0�
dense_65/kernel/Regularizer/AbsAbs6dense_65/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	�t
#dense_65/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_65/kernel/Regularizer/SumSum#dense_65/kernel/Regularizer/Abs:y:0,dense_65/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_65/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/kernel/Regularizer/mulMul*dense_65/kernel/Regularizer/mul/x:output:0(dense_65/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/kernel/Regularizer/addAddV2*dense_65/kernel/Regularizer/Const:output:0#dense_65/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_65_21797593*
_output_shapes
:	�*
dtype0�
"dense_65/kernel/Regularizer/L2LossL2Loss9dense_65/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_65/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_65/kernel/Regularizer/mul_1Mul,dense_65/kernel/Regularizer/mul_1/x:output:0+dense_65/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_65/kernel/Regularizer/add_1AddV2#dense_65/kernel/Regularizer/add:z:0%dense_65/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_65/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_65_21797595*
_output_shapes	
:�*
dtype0�
dense_65/bias/Regularizer/AbsAbs4dense_65/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_65/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_65/bias/Regularizer/SumSum!dense_65/bias/Regularizer/Abs:y:0*dense_65/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_65/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_65/bias/Regularizer/mulMul(dense_65/bias/Regularizer/mul/x:output:0&dense_65/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/addAddV2(dense_65/bias/Regularizer/Const:output:0!dense_65/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_65/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_65_21797595*
_output_shapes	
:�*
dtype0�
 dense_65/bias/Regularizer/L2LossL2Loss7dense_65/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_65/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_65/bias/Regularizer/mul_1Mul*dense_65/bias/Regularizer/mul_1/x:output:0)dense_65/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_65/bias/Regularizer/add_1AddV2!dense_65/bias/Regularizer/add:z:0#dense_65/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_66/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_66/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_66_21797635* 
_output_shapes
:
��*
dtype0�
dense_66/kernel/Regularizer/AbsAbs6dense_66/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_66/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_66/kernel/Regularizer/SumSum#dense_66/kernel/Regularizer/Abs:y:0,dense_66/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_66/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/kernel/Regularizer/mulMul*dense_66/kernel/Regularizer/mul/x:output:0(dense_66/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/kernel/Regularizer/addAddV2*dense_66/kernel/Regularizer/Const:output:0#dense_66/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_66_21797635* 
_output_shapes
:
��*
dtype0�
"dense_66/kernel/Regularizer/L2LossL2Loss9dense_66/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_66/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_66/kernel/Regularizer/mul_1Mul,dense_66/kernel/Regularizer/mul_1/x:output:0+dense_66/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_66/kernel/Regularizer/add_1AddV2#dense_66/kernel/Regularizer/add:z:0%dense_66/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_66/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_66_21797637*
_output_shapes	
:�*
dtype0�
dense_66/bias/Regularizer/AbsAbs4dense_66/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_66/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_66/bias/Regularizer/SumSum!dense_66/bias/Regularizer/Abs:y:0*dense_66/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/bias/Regularizer/mulMul(dense_66/bias/Regularizer/mul/x:output:0&dense_66/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/addAddV2(dense_66/bias/Regularizer/Const:output:0!dense_66/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_66/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_66_21797637*
_output_shapes	
:�*
dtype0�
 dense_66/bias/Regularizer/L2LossL2Loss7dense_66/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_66/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_66/bias/Regularizer/mul_1Mul*dense_66/bias/Regularizer/mul_1/x:output:0)dense_66/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/add_1AddV2!dense_66/bias/Regularizer/add:z:0#dense_66/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_67/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_67/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_67_21797677* 
_output_shapes
:
��*
dtype0�
dense_67/kernel/Regularizer/AbsAbs6dense_67/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_67/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_67/kernel/Regularizer/SumSum#dense_67/kernel/Regularizer/Abs:y:0,dense_67/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_67/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/kernel/Regularizer/mulMul*dense_67/kernel/Regularizer/mul/x:output:0(dense_67/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/kernel/Regularizer/addAddV2*dense_67/kernel/Regularizer/Const:output:0#dense_67/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_67_21797677* 
_output_shapes
:
��*
dtype0�
"dense_67/kernel/Regularizer/L2LossL2Loss9dense_67/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_67/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_67/kernel/Regularizer/mul_1Mul,dense_67/kernel/Regularizer/mul_1/x:output:0+dense_67/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_67/kernel/Regularizer/add_1AddV2#dense_67/kernel/Regularizer/add:z:0%dense_67/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_67/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_67_21797679*
_output_shapes	
:�*
dtype0�
dense_67/bias/Regularizer/AbsAbs4dense_67/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_67/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_67/bias/Regularizer/SumSum!dense_67/bias/Regularizer/Abs:y:0*dense_67/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/bias/Regularizer/mulMul(dense_67/bias/Regularizer/mul/x:output:0&dense_67/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/addAddV2(dense_67/bias/Regularizer/Const:output:0!dense_67/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_67/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_67_21797679*
_output_shapes	
:�*
dtype0�
 dense_67/bias/Regularizer/L2LossL2Loss7dense_67/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_67/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_67/bias/Regularizer/mul_1Mul*dense_67/bias/Regularizer/mul_1/x:output:0)dense_67/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/add_1AddV2!dense_67/bias/Regularizer/add:z:0#dense_67/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_68/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_68/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_68_21797719* 
_output_shapes
:
��*
dtype0�
dense_68/kernel/Regularizer/AbsAbs6dense_68/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_68/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_68/kernel/Regularizer/SumSum#dense_68/kernel/Regularizer/Abs:y:0,dense_68/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_68/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/kernel/Regularizer/mulMul*dense_68/kernel/Regularizer/mul/x:output:0(dense_68/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/kernel/Regularizer/addAddV2*dense_68/kernel/Regularizer/Const:output:0#dense_68/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_68_21797719* 
_output_shapes
:
��*
dtype0�
"dense_68/kernel/Regularizer/L2LossL2Loss9dense_68/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_68/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_68/kernel/Regularizer/mul_1Mul,dense_68/kernel/Regularizer/mul_1/x:output:0+dense_68/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_68/kernel/Regularizer/add_1AddV2#dense_68/kernel/Regularizer/add:z:0%dense_68/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_68/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_68_21797721*
_output_shapes	
:�*
dtype0�
dense_68/bias/Regularizer/AbsAbs4dense_68/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_68/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_68/bias/Regularizer/SumSum!dense_68/bias/Regularizer/Abs:y:0*dense_68/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/bias/Regularizer/mulMul(dense_68/bias/Regularizer/mul/x:output:0&dense_68/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/addAddV2(dense_68/bias/Regularizer/Const:output:0!dense_68/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_68/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_68_21797721*
_output_shapes	
:�*
dtype0�
 dense_68/bias/Regularizer/L2LossL2Loss7dense_68/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_68/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_68/bias/Regularizer/mul_1Mul*dense_68/bias/Regularizer/mul_1/x:output:0)dense_68/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/add_1AddV2!dense_68/bias/Regularizer/add:z:0#dense_68/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_69/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_69/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_69_21797761* 
_output_shapes
:
��*
dtype0�
dense_69/kernel/Regularizer/AbsAbs6dense_69/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_69/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_69/kernel/Regularizer/SumSum#dense_69/kernel/Regularizer/Abs:y:0,dense_69/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_69/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/kernel/Regularizer/mulMul*dense_69/kernel/Regularizer/mul/x:output:0(dense_69/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/kernel/Regularizer/addAddV2*dense_69/kernel/Regularizer/Const:output:0#dense_69/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_69_21797761* 
_output_shapes
:
��*
dtype0�
"dense_69/kernel/Regularizer/L2LossL2Loss9dense_69/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_69/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_69/kernel/Regularizer/mul_1Mul,dense_69/kernel/Regularizer/mul_1/x:output:0+dense_69/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_69/kernel/Regularizer/add_1AddV2#dense_69/kernel/Regularizer/add:z:0%dense_69/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_69/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_69_21797763*
_output_shapes	
:�*
dtype0�
dense_69/bias/Regularizer/AbsAbs4dense_69/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_69/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_69/bias/Regularizer/SumSum!dense_69/bias/Regularizer/Abs:y:0*dense_69/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/bias/Regularizer/mulMul(dense_69/bias/Regularizer/mul/x:output:0&dense_69/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/addAddV2(dense_69/bias/Regularizer/Const:output:0!dense_69/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_69/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_69_21797763*
_output_shapes	
:�*
dtype0�
 dense_69/bias/Regularizer/L2LossL2Loss7dense_69/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_69/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_69/bias/Regularizer/mul_1Mul*dense_69/bias/Regularizer/mul_1/x:output:0)dense_69/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/add_1AddV2!dense_69/bias/Regularizer/add:z:0#dense_69/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_70/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_70_21797803* 
_output_shapes
:
��*
dtype0�
dense_70/kernel/Regularizer/AbsAbs6dense_70/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_70/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_70/kernel/Regularizer/SumSum#dense_70/kernel/Regularizer/Abs:y:0,dense_70/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/kernel/Regularizer/addAddV2*dense_70/kernel/Regularizer/Const:output:0#dense_70/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_70_21797803* 
_output_shapes
:
��*
dtype0�
"dense_70/kernel/Regularizer/L2LossL2Loss9dense_70/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_70/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_70/kernel/Regularizer/mul_1Mul,dense_70/kernel/Regularizer/mul_1/x:output:0+dense_70/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_70/kernel/Regularizer/add_1AddV2#dense_70/kernel/Regularizer/add:z:0%dense_70/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    {
,dense_70/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_70_21797805*
_output_shapes	
:�*
dtype0�
dense_70/bias/Regularizer/AbsAbs4dense_70/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_70/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_70/bias/Regularizer/SumSum!dense_70/bias/Regularizer/Abs:y:0*dense_70/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/bias/Regularizer/mulMul(dense_70/bias/Regularizer/mul/x:output:0&dense_70/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/addAddV2(dense_70/bias/Regularizer/Const:output:0!dense_70/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ~
/dense_70/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_70_21797805*
_output_shapes	
:�*
dtype0�
 dense_70/bias/Regularizer/L2LossL2Loss7dense_70/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_70/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_70/bias/Regularizer/mul_1Mul*dense_70/bias/Regularizer/mul_1/x:output:0)dense_70/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/add_1AddV2!dense_70/bias/Regularizer/add:z:0#dense_70/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_71/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_65/StatefulPartitionedCall-^dense_65/bias/Regularizer/Abs/ReadVariableOp0^dense_65/bias/Regularizer/L2Loss/ReadVariableOp/^dense_65/kernel/Regularizer/Abs/ReadVariableOp2^dense_65/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_66/StatefulPartitionedCall-^dense_66/bias/Regularizer/Abs/ReadVariableOp0^dense_66/bias/Regularizer/L2Loss/ReadVariableOp/^dense_66/kernel/Regularizer/Abs/ReadVariableOp2^dense_66/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_67/StatefulPartitionedCall-^dense_67/bias/Regularizer/Abs/ReadVariableOp0^dense_67/bias/Regularizer/L2Loss/ReadVariableOp/^dense_67/kernel/Regularizer/Abs/ReadVariableOp2^dense_67/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_68/StatefulPartitionedCall-^dense_68/bias/Regularizer/Abs/ReadVariableOp0^dense_68/bias/Regularizer/L2Loss/ReadVariableOp/^dense_68/kernel/Regularizer/Abs/ReadVariableOp2^dense_68/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_69/StatefulPartitionedCall-^dense_69/bias/Regularizer/Abs/ReadVariableOp0^dense_69/bias/Regularizer/L2Loss/ReadVariableOp/^dense_69/kernel/Regularizer/Abs/ReadVariableOp2^dense_69/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_70/StatefulPartitionedCall-^dense_70/bias/Regularizer/Abs/ReadVariableOp0^dense_70/bias/Regularizer/L2Loss/ReadVariableOp/^dense_70/kernel/Regularizer/Abs/ReadVariableOp2^dense_70/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_71/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall2\
,dense_65/bias/Regularizer/Abs/ReadVariableOp,dense_65/bias/Regularizer/Abs/ReadVariableOp2b
/dense_65/bias/Regularizer/L2Loss/ReadVariableOp/dense_65/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_65/kernel/Regularizer/Abs/ReadVariableOp.dense_65/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp1dense_65/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2\
,dense_66/bias/Regularizer/Abs/ReadVariableOp,dense_66/bias/Regularizer/Abs/ReadVariableOp2b
/dense_66/bias/Regularizer/L2Loss/ReadVariableOp/dense_66/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_66/kernel/Regularizer/Abs/ReadVariableOp.dense_66/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2\
,dense_67/bias/Regularizer/Abs/ReadVariableOp,dense_67/bias/Regularizer/Abs/ReadVariableOp2b
/dense_67/bias/Regularizer/L2Loss/ReadVariableOp/dense_67/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_67/kernel/Regularizer/Abs/ReadVariableOp.dense_67/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp1dense_67/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2\
,dense_68/bias/Regularizer/Abs/ReadVariableOp,dense_68/bias/Regularizer/Abs/ReadVariableOp2b
/dense_68/bias/Regularizer/L2Loss/ReadVariableOp/dense_68/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_68/kernel/Regularizer/Abs/ReadVariableOp.dense_68/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_69/StatefulPartitionedCall dense_69/StatefulPartitionedCall2\
,dense_69/bias/Regularizer/Abs/ReadVariableOp,dense_69/bias/Regularizer/Abs/ReadVariableOp2b
/dense_69/bias/Regularizer/L2Loss/ReadVariableOp/dense_69/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_69/kernel/Regularizer/Abs/ReadVariableOp.dense_69/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_70/StatefulPartitionedCall dense_70/StatefulPartitionedCall2\
,dense_70/bias/Regularizer/Abs/ReadVariableOp,dense_70/bias/Regularizer/Abs/ReadVariableOp2b
/dense_70/bias/Regularizer/L2Loss/ReadVariableOp/dense_70/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_70/kernel/Regularizer/Abs/ReadVariableOp.dense_70/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_71/StatefulPartitionedCall dense_71/StatefulPartitionedCall:($
"
_user_specified_name
21797820:($
"
_user_specified_name
21797818:($
"
_user_specified_name
21797805:($
"
_user_specified_name
21797803:($
"
_user_specified_name
21797763:($
"
_user_specified_name
21797761:(
$
"
_user_specified_name
21797721:(	$
"
_user_specified_name
21797719:($
"
_user_specified_name
21797679:($
"
_user_specified_name
21797677:($
"
_user_specified_name
21797637:($
"
_user_specified_name
21797635:($
"
_user_specified_name
21797595:($
"
_user_specified_name
21797593:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�	
�
F__inference_dense_71_layer_call_and_return_conditional_losses_21797817

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_11_21799168D
5dense_70_bias_regularizer_abs_readvariableop_resource:	�
identity��,dense_70/bias/Regularizer/Abs/ReadVariableOp�/dense_70/bias/Regularizer/L2Loss/ReadVariableOpd
dense_70/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_70/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_70_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_70/bias/Regularizer/AbsAbs4dense_70/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_70/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_70/bias/Regularizer/SumSum!dense_70/bias/Regularizer/Abs:y:0*dense_70/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/bias/Regularizer/mulMul(dense_70/bias/Regularizer/mul/x:output:0&dense_70/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/addAddV2(dense_70/bias/Regularizer/Const:output:0!dense_70/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_70/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp5dense_70_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_70/bias/Regularizer/L2LossL2Loss7dense_70/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_70/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_70/bias/Regularizer/mul_1Mul*dense_70/bias/Regularizer/mul_1/x:output:0)dense_70/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/add_1AddV2!dense_70/bias/Regularizer/add:z:0#dense_70/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_70/bias/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp-^dense_70/bias/Regularizer/Abs/ReadVariableOp0^dense_70/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,dense_70/bias/Regularizer/Abs/ReadVariableOp,dense_70/bias/Regularizer/Abs/ReadVariableOp2b
/dense_70/bias/Regularizer/L2Loss/ReadVariableOp/dense_70/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
+__inference_dense_65_layer_call_fn_21798678

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_65_layer_call_and_return_conditional_losses_21797592p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798674:($
"
_user_specified_name
21798672:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_21798513
normalization_input
unknown
	unknown_0
	unknown_1:	�
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_21797546o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798509:($
"
_user_specified_name
21798507:($
"
_user_specified_name
21798505:($
"
_user_specified_name
21798503:($
"
_user_specified_name
21798501:($
"
_user_specified_name
21798499:(
$
"
_user_specified_name
21798497:(	$
"
_user_specified_name
21798495:($
"
_user_specified_name
21798493:($
"
_user_specified_name
21798491:($
"
_user_specified_name
21798489:($
"
_user_specified_name
21798487:($
"
_user_specified_name
21798485:($
"
_user_specified_name
21798483:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
__inference_loss_fn_8_21799117K
7dense_69_kernel_regularizer_abs_readvariableop_resource:
��
identity��.dense_69/kernel/Regularizer/Abs/ReadVariableOp�1dense_69/kernel/Regularizer/L2Loss/ReadVariableOpf
!dense_69/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_69/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_69_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_69/kernel/Regularizer/AbsAbs6dense_69/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_69/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_69/kernel/Regularizer/SumSum#dense_69/kernel/Regularizer/Abs:y:0,dense_69/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_69/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/kernel/Regularizer/mulMul*dense_69/kernel/Regularizer/mul/x:output:0(dense_69/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/kernel/Regularizer/addAddV2*dense_69/kernel/Regularizer/Const:output:0#dense_69/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp7dense_69_kernel_regularizer_abs_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_69/kernel/Regularizer/L2LossL2Loss9dense_69/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_69/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_69/kernel/Regularizer/mul_1Mul,dense_69/kernel/Regularizer/mul_1/x:output:0+dense_69/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_69/kernel/Regularizer/add_1AddV2#dense_69/kernel/Regularizer/add:z:0%dense_69/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_69/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp/^dense_69/kernel/Regularizer/Abs/ReadVariableOp2^dense_69/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_69/kernel/Regularizer/Abs/ReadVariableOp.dense_69/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp1dense_69/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�2
!__inference__traced_save_21799534
file_prefix)
read_disablecopyonread_mean:/
!read_1_disablecopyonread_variance:*
 read_2_disablecopyonread_count_5:	 ;
(read_3_disablecopyonread_dense_65_kernel:	�5
&read_4_disablecopyonread_dense_65_bias:	�<
(read_5_disablecopyonread_dense_66_kernel:
��5
&read_6_disablecopyonread_dense_66_bias:	�<
(read_7_disablecopyonread_dense_67_kernel:
��5
&read_8_disablecopyonread_dense_67_bias:	�<
(read_9_disablecopyonread_dense_68_kernel:
��6
'read_10_disablecopyonread_dense_68_bias:	�=
)read_11_disablecopyonread_dense_69_kernel:
��6
'read_12_disablecopyonread_dense_69_bias:	�=
)read_13_disablecopyonread_dense_70_kernel:
��6
'read_14_disablecopyonread_dense_70_bias:	�<
)read_15_disablecopyonread_dense_71_kernel:	�5
'read_16_disablecopyonread_dense_71_bias:-
#read_17_disablecopyonread_iteration:	 9
/read_18_disablecopyonread_current_learning_rate: C
0read_19_disablecopyonread_adam_m_dense_65_kernel:	�C
0read_20_disablecopyonread_adam_v_dense_65_kernel:	�=
.read_21_disablecopyonread_adam_m_dense_65_bias:	�=
.read_22_disablecopyonread_adam_v_dense_65_bias:	�D
0read_23_disablecopyonread_adam_m_dense_66_kernel:
��D
0read_24_disablecopyonread_adam_v_dense_66_kernel:
��=
.read_25_disablecopyonread_adam_m_dense_66_bias:	�=
.read_26_disablecopyonread_adam_v_dense_66_bias:	�D
0read_27_disablecopyonread_adam_m_dense_67_kernel:
��D
0read_28_disablecopyonread_adam_v_dense_67_kernel:
��=
.read_29_disablecopyonread_adam_m_dense_67_bias:	�=
.read_30_disablecopyonread_adam_v_dense_67_bias:	�D
0read_31_disablecopyonread_adam_m_dense_68_kernel:
��D
0read_32_disablecopyonread_adam_v_dense_68_kernel:
��=
.read_33_disablecopyonread_adam_m_dense_68_bias:	�=
.read_34_disablecopyonread_adam_v_dense_68_bias:	�D
0read_35_disablecopyonread_adam_m_dense_69_kernel:
��D
0read_36_disablecopyonread_adam_v_dense_69_kernel:
��=
.read_37_disablecopyonread_adam_m_dense_69_bias:	�=
.read_38_disablecopyonread_adam_v_dense_69_bias:	�D
0read_39_disablecopyonread_adam_m_dense_70_kernel:
��D
0read_40_disablecopyonread_adam_v_dense_70_kernel:
��=
.read_41_disablecopyonread_adam_m_dense_70_bias:	�=
.read_42_disablecopyonread_adam_v_dense_70_bias:	�C
0read_43_disablecopyonread_adam_m_dense_71_kernel:	�C
0read_44_disablecopyonread_adam_v_dense_71_kernel:	�<
.read_45_disablecopyonread_adam_m_dense_71_bias:<
.read_46_disablecopyonread_adam_v_dense_71_bias:+
!read_47_disablecopyonread_total_4: +
!read_48_disablecopyonread_count_4: +
!read_49_disablecopyonread_total_3: +
!read_50_disablecopyonread_count_3: +
!read_51_disablecopyonread_total_2: +
!read_52_disablecopyonread_count_2: +
!read_53_disablecopyonread_total_1: +
!read_54_disablecopyonread_count_1: )
read_55_disablecopyonread_total: )
read_56_disablecopyonread_count: 
savev2_const_2
identity_115��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: m
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_mean"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpread_disablecopyonread_mean^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:u
Read_1/DisableCopyOnReadDisableCopyOnRead!read_1_disablecopyonread_variance"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp!read_1_disablecopyonread_variance^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_2/DisableCopyOnReadDisableCopyOnRead read_2_disablecopyonread_count_5"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp read_2_disablecopyonread_count_5^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_3/DisableCopyOnReadDisableCopyOnRead(read_3_disablecopyonread_dense_65_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp(read_3_disablecopyonread_dense_65_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_dense_65_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_dense_65_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_dense_66_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_dense_66_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_dense_66_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_dense_66_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_67_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_67_kernel^Read_7/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_dense_67_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_dense_67_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_9/DisableCopyOnReadDisableCopyOnRead(read_9_disablecopyonread_dense_68_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp(read_9_disablecopyonread_dense_68_kernel^Read_9/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_dense_68_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_dense_68_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_11/DisableCopyOnReadDisableCopyOnRead)read_11_disablecopyonread_dense_69_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp)read_11_disablecopyonread_dense_69_kernel^Read_11/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_12/DisableCopyOnReadDisableCopyOnRead'read_12_disablecopyonread_dense_69_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp'read_12_disablecopyonread_dense_69_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_13/DisableCopyOnReadDisableCopyOnRead)read_13_disablecopyonread_dense_70_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp)read_13_disablecopyonread_dense_70_kernel^Read_13/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_14/DisableCopyOnReadDisableCopyOnRead'read_14_disablecopyonread_dense_70_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp'read_14_disablecopyonread_dense_70_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_15/DisableCopyOnReadDisableCopyOnRead)read_15_disablecopyonread_dense_71_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp)read_15_disablecopyonread_dense_71_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	�|
Read_16/DisableCopyOnReadDisableCopyOnRead'read_16_disablecopyonread_dense_71_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp'read_16_disablecopyonread_dense_71_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_17/DisableCopyOnReadDisableCopyOnRead#read_17_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp#read_17_disablecopyonread_iteration^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0	*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_current_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_current_learning_rate^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_m_dense_65_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_m_dense_65_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_adam_v_dense_65_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_adam_v_dense_65_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_adam_m_dense_65_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_adam_m_dense_65_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_v_dense_65_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_v_dense_65_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_m_dense_66_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_m_dense_66_kernel^Read_23/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_adam_v_dense_66_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_adam_v_dense_66_kernel^Read_24/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_m_dense_66_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_m_dense_66_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_v_dense_66_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_v_dense_66_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_m_dense_67_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_m_dense_67_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_28/DisableCopyOnReadDisableCopyOnRead0read_28_disablecopyonread_adam_v_dense_67_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp0read_28_disablecopyonread_adam_v_dense_67_kernel^Read_28/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_m_dense_67_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_m_dense_67_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_v_dense_67_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_v_dense_67_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_m_dense_68_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_m_dense_68_kernel^Read_31/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_32/DisableCopyOnReadDisableCopyOnRead0read_32_disablecopyonread_adam_v_dense_68_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp0read_32_disablecopyonread_adam_v_dense_68_kernel^Read_32/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_m_dense_68_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_m_dense_68_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead.read_34_disablecopyonread_adam_v_dense_68_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp.read_34_disablecopyonread_adam_v_dense_68_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_m_dense_69_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_m_dense_69_kernel^Read_35/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_36/DisableCopyOnReadDisableCopyOnRead0read_36_disablecopyonread_adam_v_dense_69_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp0read_36_disablecopyonread_adam_v_dense_69_kernel^Read_36/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_m_dense_69_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_m_dense_69_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_38/DisableCopyOnReadDisableCopyOnRead.read_38_disablecopyonread_adam_v_dense_69_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp.read_38_disablecopyonread_adam_v_dense_69_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_m_dense_70_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_m_dense_70_kernel^Read_39/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_40/DisableCopyOnReadDisableCopyOnRead0read_40_disablecopyonread_adam_v_dense_70_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp0read_40_disablecopyonread_adam_v_dense_70_kernel^Read_40/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_m_dense_70_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_m_dense_70_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead.read_42_disablecopyonread_adam_v_dense_70_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp.read_42_disablecopyonread_adam_v_dense_70_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_m_dense_71_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_m_dense_71_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_44/DisableCopyOnReadDisableCopyOnRead0read_44_disablecopyonread_adam_v_dense_71_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp0read_44_disablecopyonread_adam_v_dense_71_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_m_dense_71_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_m_dense_71_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnRead.read_46_disablecopyonread_adam_v_dense_71_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp.read_46_disablecopyonread_adam_v_dense_71_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_47/DisableCopyOnReadDisableCopyOnRead!read_47_disablecopyonread_total_4"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp!read_47_disablecopyonread_total_4^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_48/DisableCopyOnReadDisableCopyOnRead!read_48_disablecopyonread_count_4"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp!read_48_disablecopyonread_count_4^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_49/DisableCopyOnReadDisableCopyOnRead!read_49_disablecopyonread_total_3"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp!read_49_disablecopyonread_total_3^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_50/DisableCopyOnReadDisableCopyOnRead!read_50_disablecopyonread_count_3"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp!read_50_disablecopyonread_count_3^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_51/DisableCopyOnReadDisableCopyOnRead!read_51_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp!read_51_disablecopyonread_total_2^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_52/DisableCopyOnReadDisableCopyOnRead!read_52_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp!read_52_disablecopyonread_count_2^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_53/DisableCopyOnReadDisableCopyOnRead!read_53_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp!read_53_disablecopyonread_total_1^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_54/DisableCopyOnReadDisableCopyOnRead!read_54_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp!read_54_disablecopyonread_count_1^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_55/DisableCopyOnReadDisableCopyOnReadread_55_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpread_55_disablecopyonread_total^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_56/DisableCopyOnReadDisableCopyOnReadread_56_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOpread_56_disablecopyonread_count^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0savev2_const_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *H
dtypes>
<2:		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_114Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_115IdentityIdentity_114:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_115Identity_115:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:?:;

_output_shapes
: 
!
_user_specified_name	Const_2:%9!

_user_specified_namecount:%8!

_user_specified_nametotal:'7#
!
_user_specified_name	count_1:'6#
!
_user_specified_name	total_1:'5#
!
_user_specified_name	count_2:'4#
!
_user_specified_name	total_2:'3#
!
_user_specified_name	count_3:'2#
!
_user_specified_name	total_3:'1#
!
_user_specified_name	count_4:'0#
!
_user_specified_name	total_4:4/0
.
_user_specified_nameAdam/v/dense_71/bias:4.0
.
_user_specified_nameAdam/m/dense_71/bias:6-2
0
_user_specified_nameAdam/v/dense_71/kernel:6,2
0
_user_specified_nameAdam/m/dense_71/kernel:4+0
.
_user_specified_nameAdam/v/dense_70/bias:4*0
.
_user_specified_nameAdam/m/dense_70/bias:6)2
0
_user_specified_nameAdam/v/dense_70/kernel:6(2
0
_user_specified_nameAdam/m/dense_70/kernel:4'0
.
_user_specified_nameAdam/v/dense_69/bias:4&0
.
_user_specified_nameAdam/m/dense_69/bias:6%2
0
_user_specified_nameAdam/v/dense_69/kernel:6$2
0
_user_specified_nameAdam/m/dense_69/kernel:4#0
.
_user_specified_nameAdam/v/dense_68/bias:4"0
.
_user_specified_nameAdam/m/dense_68/bias:6!2
0
_user_specified_nameAdam/v/dense_68/kernel:6 2
0
_user_specified_nameAdam/m/dense_68/kernel:40
.
_user_specified_nameAdam/v/dense_67/bias:40
.
_user_specified_nameAdam/m/dense_67/bias:62
0
_user_specified_nameAdam/v/dense_67/kernel:62
0
_user_specified_nameAdam/m/dense_67/kernel:40
.
_user_specified_nameAdam/v/dense_66/bias:40
.
_user_specified_nameAdam/m/dense_66/bias:62
0
_user_specified_nameAdam/v/dense_66/kernel:62
0
_user_specified_nameAdam/m/dense_66/kernel:40
.
_user_specified_nameAdam/v/dense_65/bias:40
.
_user_specified_nameAdam/m/dense_65/bias:62
0
_user_specified_nameAdam/v/dense_65/kernel:62
0
_user_specified_nameAdam/m/dense_65/kernel:51
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_71/bias:/+
)
_user_specified_namedense_71/kernel:-)
'
_user_specified_namedense_70/bias:/+
)
_user_specified_namedense_70/kernel:-)
'
_user_specified_namedense_69/bias:/+
)
_user_specified_namedense_69/kernel:-)
'
_user_specified_namedense_68/bias:/
+
)
_user_specified_namedense_68/kernel:-	)
'
_user_specified_namedense_67/bias:/+
)
_user_specified_namedense_67/kernel:-)
'
_user_specified_namedense_66/bias:/+
)
_user_specified_namedense_66/kernel:-)
'
_user_specified_namedense_65/bias:/+
)
_user_specified_namedense_65/kernel:'#
!
_user_specified_name	count_5:($
"
_user_specified_name
variance:$ 

_user_specified_namemean:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
__inference_loss_fn_9_21799134D
5dense_69_bias_regularizer_abs_readvariableop_resource:	�
identity��,dense_69/bias/Regularizer/Abs/ReadVariableOp�/dense_69/bias/Regularizer/L2Loss/ReadVariableOpd
dense_69/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_69/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_69_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_69/bias/Regularizer/AbsAbs4dense_69/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_69/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_69/bias/Regularizer/SumSum!dense_69/bias/Regularizer/Abs:y:0*dense_69/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_69/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_69/bias/Regularizer/mulMul(dense_69/bias/Regularizer/mul/x:output:0&dense_69/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/addAddV2(dense_69/bias/Regularizer/Const:output:0!dense_69/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_69/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp5dense_69_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_69/bias/Regularizer/L2LossL2Loss7dense_69/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_69/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_69/bias/Regularizer/mul_1Mul*dense_69/bias/Regularizer/mul_1/x:output:0)dense_69/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_69/bias/Regularizer/add_1AddV2!dense_69/bias/Regularizer/add:z:0#dense_69/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_69/bias/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp-^dense_69/bias/Regularizer/Abs/ReadVariableOp0^dense_69/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,dense_69/bias/Regularizer/Abs/ReadVariableOp,dense_69/bias/Regularizer/Abs/ReadVariableOp2b
/dense_69/bias/Regularizer/L2Loss/ReadVariableOp/dense_69/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
__inference_loss_fn_5_21799066D
5dense_67_bias_regularizer_abs_readvariableop_resource:	�
identity��,dense_67/bias/Regularizer/Abs/ReadVariableOp�/dense_67/bias/Regularizer/L2Loss/ReadVariableOpd
dense_67/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_67/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_67_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_67/bias/Regularizer/AbsAbs4dense_67/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_67/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_67/bias/Regularizer/SumSum!dense_67/bias/Regularizer/Abs:y:0*dense_67/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_67/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_67/bias/Regularizer/mulMul(dense_67/bias/Regularizer/mul/x:output:0&dense_67/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/addAddV2(dense_67/bias/Regularizer/Const:output:0!dense_67/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_67/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp5dense_67_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_67/bias/Regularizer/L2LossL2Loss7dense_67/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_67/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_67/bias/Regularizer/mul_1Mul*dense_67/bias/Regularizer/mul_1/x:output:0)dense_67/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_67/bias/Regularizer/add_1AddV2!dense_67/bias/Regularizer/add:z:0#dense_67/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_67/bias/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp-^dense_67/bias/Regularizer/Abs/ReadVariableOp0^dense_67/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,dense_67/bias/Regularizer/Abs/ReadVariableOp,dense_67/bias/Regularizer/Abs/ReadVariableOp2b
/dense_67/bias/Regularizer/L2Loss/ReadVariableOp/dense_67/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
+__inference_dense_68_layer_call_fn_21798816

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_68_layer_call_and_return_conditional_losses_21797718p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798812:($
"
_user_specified_name
21798810:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
F__inference_dense_66_layer_call_and_return_conditional_losses_21797634

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_66/bias/Regularizer/Abs/ReadVariableOp�/dense_66/bias/Regularizer/L2Loss/ReadVariableOp�.dense_66/kernel/Regularizer/Abs/ReadVariableOp�1dense_66/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_66/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_66/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_66/kernel/Regularizer/AbsAbs6dense_66/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_66/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_66/kernel/Regularizer/SumSum#dense_66/kernel/Regularizer/Abs:y:0,dense_66/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_66/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/kernel/Regularizer/mulMul*dense_66/kernel/Regularizer/mul/x:output:0(dense_66/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/kernel/Regularizer/addAddV2*dense_66/kernel/Regularizer/Const:output:0#dense_66/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_66/kernel/Regularizer/L2LossL2Loss9dense_66/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_66/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_66/kernel/Regularizer/mul_1Mul,dense_66/kernel/Regularizer/mul_1/x:output:0+dense_66/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_66/kernel/Regularizer/add_1AddV2#dense_66/kernel/Regularizer/add:z:0%dense_66/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_66/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_66/bias/Regularizer/AbsAbs4dense_66/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_66/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_66/bias/Regularizer/SumSum!dense_66/bias/Regularizer/Abs:y:0*dense_66/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/bias/Regularizer/mulMul(dense_66/bias/Regularizer/mul/x:output:0&dense_66/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/addAddV2(dense_66/bias/Regularizer/Const:output:0!dense_66/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_66/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_66/bias/Regularizer/L2LossL2Loss7dense_66/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_66/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_66/bias/Regularizer/mul_1Mul*dense_66/bias/Regularizer/mul_1/x:output:0)dense_66/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/add_1AddV2!dense_66/bias/Regularizer/add:z:0#dense_66/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_66/bias/Regularizer/Abs/ReadVariableOp0^dense_66/bias/Regularizer/L2Loss/ReadVariableOp/^dense_66/kernel/Regularizer/Abs/ReadVariableOp2^dense_66/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_66/bias/Regularizer/Abs/ReadVariableOp,dense_66/bias/Regularizer/Abs/ReadVariableOp2b
/dense_66/bias/Regularizer/L2Loss/ReadVariableOp/dense_66/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_66/kernel/Regularizer/Abs/ReadVariableOp.dense_66/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
F__inference_dense_68_layer_call_and_return_conditional_losses_21797718

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_68/bias/Regularizer/Abs/ReadVariableOp�/dense_68/bias/Regularizer/L2Loss/ReadVariableOp�.dense_68/kernel/Regularizer/Abs/ReadVariableOp�1dense_68/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_68/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_68/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_68/kernel/Regularizer/AbsAbs6dense_68/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_68/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_68/kernel/Regularizer/SumSum#dense_68/kernel/Regularizer/Abs:y:0,dense_68/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_68/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/kernel/Regularizer/mulMul*dense_68/kernel/Regularizer/mul/x:output:0(dense_68/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/kernel/Regularizer/addAddV2*dense_68/kernel/Regularizer/Const:output:0#dense_68/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_68/kernel/Regularizer/L2LossL2Loss9dense_68/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_68/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_68/kernel/Regularizer/mul_1Mul,dense_68/kernel/Regularizer/mul_1/x:output:0+dense_68/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_68/kernel/Regularizer/add_1AddV2#dense_68/kernel/Regularizer/add:z:0%dense_68/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_68/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_68/bias/Regularizer/AbsAbs4dense_68/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_68/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_68/bias/Regularizer/SumSum!dense_68/bias/Regularizer/Abs:y:0*dense_68/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_68/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_68/bias/Regularizer/mulMul(dense_68/bias/Regularizer/mul/x:output:0&dense_68/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/addAddV2(dense_68/bias/Regularizer/Const:output:0!dense_68/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_68/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_68/bias/Regularizer/L2LossL2Loss7dense_68/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_68/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_68/bias/Regularizer/mul_1Mul*dense_68/bias/Regularizer/mul_1/x:output:0)dense_68/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_68/bias/Regularizer/add_1AddV2!dense_68/bias/Regularizer/add:z:0#dense_68/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_68/bias/Regularizer/Abs/ReadVariableOp0^dense_68/bias/Regularizer/L2Loss/ReadVariableOp/^dense_68/kernel/Regularizer/Abs/ReadVariableOp2^dense_68/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_68/bias/Regularizer/Abs/ReadVariableOp,dense_68/bias/Regularizer/Abs/ReadVariableOp2b
/dense_68/bias/Regularizer/L2Loss/ReadVariableOp/dense_68/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_68/kernel/Regularizer/Abs/ReadVariableOp.dense_68/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp1dense_68/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�!
$__inference__traced_restore_21799714
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:$
assignvariableop_2_count_5:	 5
"assignvariableop_3_dense_65_kernel:	�/
 assignvariableop_4_dense_65_bias:	�6
"assignvariableop_5_dense_66_kernel:
��/
 assignvariableop_6_dense_66_bias:	�6
"assignvariableop_7_dense_67_kernel:
��/
 assignvariableop_8_dense_67_bias:	�6
"assignvariableop_9_dense_68_kernel:
��0
!assignvariableop_10_dense_68_bias:	�7
#assignvariableop_11_dense_69_kernel:
��0
!assignvariableop_12_dense_69_bias:	�7
#assignvariableop_13_dense_70_kernel:
��0
!assignvariableop_14_dense_70_bias:	�6
#assignvariableop_15_dense_71_kernel:	�/
!assignvariableop_16_dense_71_bias:'
assignvariableop_17_iteration:	 3
)assignvariableop_18_current_learning_rate: =
*assignvariableop_19_adam_m_dense_65_kernel:	�=
*assignvariableop_20_adam_v_dense_65_kernel:	�7
(assignvariableop_21_adam_m_dense_65_bias:	�7
(assignvariableop_22_adam_v_dense_65_bias:	�>
*assignvariableop_23_adam_m_dense_66_kernel:
��>
*assignvariableop_24_adam_v_dense_66_kernel:
��7
(assignvariableop_25_adam_m_dense_66_bias:	�7
(assignvariableop_26_adam_v_dense_66_bias:	�>
*assignvariableop_27_adam_m_dense_67_kernel:
��>
*assignvariableop_28_adam_v_dense_67_kernel:
��7
(assignvariableop_29_adam_m_dense_67_bias:	�7
(assignvariableop_30_adam_v_dense_67_bias:	�>
*assignvariableop_31_adam_m_dense_68_kernel:
��>
*assignvariableop_32_adam_v_dense_68_kernel:
��7
(assignvariableop_33_adam_m_dense_68_bias:	�7
(assignvariableop_34_adam_v_dense_68_bias:	�>
*assignvariableop_35_adam_m_dense_69_kernel:
��>
*assignvariableop_36_adam_v_dense_69_kernel:
��7
(assignvariableop_37_adam_m_dense_69_bias:	�7
(assignvariableop_38_adam_v_dense_69_bias:	�>
*assignvariableop_39_adam_m_dense_70_kernel:
��>
*assignvariableop_40_adam_v_dense_70_kernel:
��7
(assignvariableop_41_adam_m_dense_70_bias:	�7
(assignvariableop_42_adam_v_dense_70_bias:	�=
*assignvariableop_43_adam_m_dense_71_kernel:	�=
*assignvariableop_44_adam_v_dense_71_kernel:	�6
(assignvariableop_45_adam_m_dense_71_bias:6
(assignvariableop_46_adam_v_dense_71_bias:%
assignvariableop_47_total_4: %
assignvariableop_48_count_4: %
assignvariableop_49_total_3: %
assignvariableop_50_count_3: %
assignvariableop_51_total_2: %
assignvariableop_52_count_2: %
assignvariableop_53_total_1: %
assignvariableop_54_count_1: #
assignvariableop_55_total: #
assignvariableop_56_count: 
identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_count_5Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_65_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_65_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_66_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_66_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_67_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_67_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_68_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_68_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_69_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_69_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_70_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_70_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_71_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_71_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_iterationIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_current_learning_rateIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_m_dense_65_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_v_dense_65_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_m_dense_65_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_v_dense_65_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_m_dense_66_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_v_dense_66_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_m_dense_66_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_v_dense_66_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_m_dense_67_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_v_dense_67_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_m_dense_67_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_v_dense_67_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_m_dense_68_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_v_dense_68_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_m_dense_68_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_v_dense_68_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_m_dense_69_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_v_dense_69_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_m_dense_69_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_v_dense_69_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_m_dense_70_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_v_dense_70_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_m_dense_70_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_v_dense_70_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_m_dense_71_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_v_dense_71_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_m_dense_71_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_v_dense_71_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_4Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_4Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_3Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_3Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_2Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_2Identity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_1Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_1Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpassignvariableop_55_totalIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_countIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_58Identity_58:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%9!

_user_specified_namecount:%8!

_user_specified_nametotal:'7#
!
_user_specified_name	count_1:'6#
!
_user_specified_name	total_1:'5#
!
_user_specified_name	count_2:'4#
!
_user_specified_name	total_2:'3#
!
_user_specified_name	count_3:'2#
!
_user_specified_name	total_3:'1#
!
_user_specified_name	count_4:'0#
!
_user_specified_name	total_4:4/0
.
_user_specified_nameAdam/v/dense_71/bias:4.0
.
_user_specified_nameAdam/m/dense_71/bias:6-2
0
_user_specified_nameAdam/v/dense_71/kernel:6,2
0
_user_specified_nameAdam/m/dense_71/kernel:4+0
.
_user_specified_nameAdam/v/dense_70/bias:4*0
.
_user_specified_nameAdam/m/dense_70/bias:6)2
0
_user_specified_nameAdam/v/dense_70/kernel:6(2
0
_user_specified_nameAdam/m/dense_70/kernel:4'0
.
_user_specified_nameAdam/v/dense_69/bias:4&0
.
_user_specified_nameAdam/m/dense_69/bias:6%2
0
_user_specified_nameAdam/v/dense_69/kernel:6$2
0
_user_specified_nameAdam/m/dense_69/kernel:4#0
.
_user_specified_nameAdam/v/dense_68/bias:4"0
.
_user_specified_nameAdam/m/dense_68/bias:6!2
0
_user_specified_nameAdam/v/dense_68/kernel:6 2
0
_user_specified_nameAdam/m/dense_68/kernel:40
.
_user_specified_nameAdam/v/dense_67/bias:40
.
_user_specified_nameAdam/m/dense_67/bias:62
0
_user_specified_nameAdam/v/dense_67/kernel:62
0
_user_specified_nameAdam/m/dense_67/kernel:40
.
_user_specified_nameAdam/v/dense_66/bias:40
.
_user_specified_nameAdam/m/dense_66/bias:62
0
_user_specified_nameAdam/v/dense_66/kernel:62
0
_user_specified_nameAdam/m/dense_66/kernel:40
.
_user_specified_nameAdam/v/dense_65/bias:40
.
_user_specified_nameAdam/m/dense_65/bias:62
0
_user_specified_nameAdam/v/dense_65/kernel:62
0
_user_specified_nameAdam/m/dense_65/kernel:51
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_71/bias:/+
)
_user_specified_namedense_71/kernel:-)
'
_user_specified_namedense_70/bias:/+
)
_user_specified_namedense_70/kernel:-)
'
_user_specified_namedense_69/bias:/+
)
_user_specified_namedense_69/kernel:-)
'
_user_specified_namedense_68/bias:/
+
)
_user_specified_namedense_68/kernel:-	)
'
_user_specified_namedense_67/bias:/+
)
_user_specified_namedense_67/kernel:-)
'
_user_specified_namedense_66/bias:/+
)
_user_specified_namedense_66/kernel:-)
'
_user_specified_namedense_65/bias:/+
)
_user_specified_namedense_65/kernel:'#
!
_user_specified_name	count_5:($
"
_user_specified_name
variance:$ 

_user_specified_namemean:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
F__inference_dense_71_layer_call_and_return_conditional_losses_21798964

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_21799032D
5dense_66_bias_regularizer_abs_readvariableop_resource:	�
identity��,dense_66/bias/Regularizer/Abs/ReadVariableOp�/dense_66/bias/Regularizer/L2Loss/ReadVariableOpd
dense_66/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_66/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_66_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_66/bias/Regularizer/AbsAbs4dense_66/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_66/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_66/bias/Regularizer/SumSum!dense_66/bias/Regularizer/Abs:y:0*dense_66/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/bias/Regularizer/mulMul(dense_66/bias/Regularizer/mul/x:output:0&dense_66/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/addAddV2(dense_66/bias/Regularizer/Const:output:0!dense_66/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_66/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOp5dense_66_bias_regularizer_abs_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_66/bias/Regularizer/L2LossL2Loss7dense_66/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_66/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_66/bias/Regularizer/mul_1Mul*dense_66/bias/Regularizer/mul_1/x:output:0)dense_66/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/add_1AddV2!dense_66/bias/Regularizer/add:z:0#dense_66/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_66/bias/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp-^dense_66/bias/Regularizer/Abs/ReadVariableOp0^dense_66/bias/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,dense_66/bias/Regularizer/Abs/ReadVariableOp,dense_66/bias/Regularizer/Abs/ReadVariableOp2b
/dense_66/bias/Regularizer/L2Loss/ReadVariableOp/dense_66/bias/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
+__inference_dense_66_layer_call_fn_21798724

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_66_layer_call_and_return_conditional_losses_21797634p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798720:($
"
_user_specified_name
21798718:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_69_layer_call_fn_21798862

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_69_layer_call_and_return_conditional_losses_21797760p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798858:($
"
_user_specified_name
21798856:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�Y
�
#__inference__wrapped_model_21797546
normalization_input%
!sequential_13_normalization_sub_y&
"sequential_13_normalization_sqrt_xH
5sequential_13_dense_65_matmul_readvariableop_resource:	�E
6sequential_13_dense_65_biasadd_readvariableop_resource:	�I
5sequential_13_dense_66_matmul_readvariableop_resource:
��E
6sequential_13_dense_66_biasadd_readvariableop_resource:	�I
5sequential_13_dense_67_matmul_readvariableop_resource:
��E
6sequential_13_dense_67_biasadd_readvariableop_resource:	�I
5sequential_13_dense_68_matmul_readvariableop_resource:
��E
6sequential_13_dense_68_biasadd_readvariableop_resource:	�I
5sequential_13_dense_69_matmul_readvariableop_resource:
��E
6sequential_13_dense_69_biasadd_readvariableop_resource:	�I
5sequential_13_dense_70_matmul_readvariableop_resource:
��E
6sequential_13_dense_70_biasadd_readvariableop_resource:	�H
5sequential_13_dense_71_matmul_readvariableop_resource:	�D
6sequential_13_dense_71_biasadd_readvariableop_resource:
identity��-sequential_13/dense_65/BiasAdd/ReadVariableOp�,sequential_13/dense_65/MatMul/ReadVariableOp�-sequential_13/dense_66/BiasAdd/ReadVariableOp�,sequential_13/dense_66/MatMul/ReadVariableOp�-sequential_13/dense_67/BiasAdd/ReadVariableOp�,sequential_13/dense_67/MatMul/ReadVariableOp�-sequential_13/dense_68/BiasAdd/ReadVariableOp�,sequential_13/dense_68/MatMul/ReadVariableOp�-sequential_13/dense_69/BiasAdd/ReadVariableOp�,sequential_13/dense_69/MatMul/ReadVariableOp�-sequential_13/dense_70/BiasAdd/ReadVariableOp�,sequential_13/dense_70/MatMul/ReadVariableOp�-sequential_13/dense_71/BiasAdd/ReadVariableOp�,sequential_13/dense_71/MatMul/ReadVariableOp�
sequential_13/normalization/subSubnormalization_input!sequential_13_normalization_sub_y*
T0*'
_output_shapes
:���������u
 sequential_13/normalization/SqrtSqrt"sequential_13_normalization_sqrt_x*
T0*
_output_shapes

:j
%sequential_13/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
#sequential_13/normalization/MaximumMaximum$sequential_13/normalization/Sqrt:y:0.sequential_13/normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
#sequential_13/normalization/truedivRealDiv#sequential_13/normalization/sub:z:0'sequential_13/normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
,sequential_13/dense_65/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_65_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_13/dense_65/MatMulMatMul'sequential_13/normalization/truediv:z:04sequential_13/dense_65/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_13/dense_65/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_65_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_13/dense_65/BiasAddBiasAdd'sequential_13/dense_65/MatMul:product:05sequential_13/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_13/dense_65/ReluRelu'sequential_13/dense_65/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_13/dense_66/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_66_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_13/dense_66/MatMulMatMul)sequential_13/dense_65/Relu:activations:04sequential_13/dense_66/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_13/dense_66/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_66_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_13/dense_66/BiasAddBiasAdd'sequential_13/dense_66/MatMul:product:05sequential_13/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_13/dense_66/ReluRelu'sequential_13/dense_66/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_13/dense_67/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_67_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_13/dense_67/MatMulMatMul)sequential_13/dense_66/Relu:activations:04sequential_13/dense_67/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_13/dense_67/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_67_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_13/dense_67/BiasAddBiasAdd'sequential_13/dense_67/MatMul:product:05sequential_13/dense_67/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_13/dense_67/ReluRelu'sequential_13/dense_67/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_13/dense_68/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_68_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_13/dense_68/MatMulMatMul)sequential_13/dense_67/Relu:activations:04sequential_13/dense_68/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_13/dense_68/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_68_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_13/dense_68/BiasAddBiasAdd'sequential_13/dense_68/MatMul:product:05sequential_13/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_13/dense_68/ReluRelu'sequential_13/dense_68/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_13/dense_69/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_69_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_13/dense_69/MatMulMatMul)sequential_13/dense_68/Relu:activations:04sequential_13/dense_69/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_13/dense_69/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_69_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_13/dense_69/BiasAddBiasAdd'sequential_13/dense_69/MatMul:product:05sequential_13/dense_69/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_13/dense_69/ReluRelu'sequential_13/dense_69/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_13/dense_70/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_70_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_13/dense_70/MatMulMatMul)sequential_13/dense_69/Relu:activations:04sequential_13/dense_70/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_13/dense_70/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_70_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_13/dense_70/BiasAddBiasAdd'sequential_13/dense_70/MatMul:product:05sequential_13/dense_70/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_13/dense_70/ReluRelu'sequential_13/dense_70/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_13/dense_71/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_71_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_13/dense_71/MatMulMatMul)sequential_13/dense_70/Relu:activations:04sequential_13/dense_71/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_13/dense_71/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_71_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_13/dense_71/BiasAddBiasAdd'sequential_13/dense_71/MatMul:product:05sequential_13/dense_71/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_13/dense_71/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_13/dense_65/BiasAdd/ReadVariableOp-^sequential_13/dense_65/MatMul/ReadVariableOp.^sequential_13/dense_66/BiasAdd/ReadVariableOp-^sequential_13/dense_66/MatMul/ReadVariableOp.^sequential_13/dense_67/BiasAdd/ReadVariableOp-^sequential_13/dense_67/MatMul/ReadVariableOp.^sequential_13/dense_68/BiasAdd/ReadVariableOp-^sequential_13/dense_68/MatMul/ReadVariableOp.^sequential_13/dense_69/BiasAdd/ReadVariableOp-^sequential_13/dense_69/MatMul/ReadVariableOp.^sequential_13/dense_70/BiasAdd/ReadVariableOp-^sequential_13/dense_70/MatMul/ReadVariableOp.^sequential_13/dense_71/BiasAdd/ReadVariableOp-^sequential_13/dense_71/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 2^
-sequential_13/dense_65/BiasAdd/ReadVariableOp-sequential_13/dense_65/BiasAdd/ReadVariableOp2\
,sequential_13/dense_65/MatMul/ReadVariableOp,sequential_13/dense_65/MatMul/ReadVariableOp2^
-sequential_13/dense_66/BiasAdd/ReadVariableOp-sequential_13/dense_66/BiasAdd/ReadVariableOp2\
,sequential_13/dense_66/MatMul/ReadVariableOp,sequential_13/dense_66/MatMul/ReadVariableOp2^
-sequential_13/dense_67/BiasAdd/ReadVariableOp-sequential_13/dense_67/BiasAdd/ReadVariableOp2\
,sequential_13/dense_67/MatMul/ReadVariableOp,sequential_13/dense_67/MatMul/ReadVariableOp2^
-sequential_13/dense_68/BiasAdd/ReadVariableOp-sequential_13/dense_68/BiasAdd/ReadVariableOp2\
,sequential_13/dense_68/MatMul/ReadVariableOp,sequential_13/dense_68/MatMul/ReadVariableOp2^
-sequential_13/dense_69/BiasAdd/ReadVariableOp-sequential_13/dense_69/BiasAdd/ReadVariableOp2\
,sequential_13/dense_69/MatMul/ReadVariableOp,sequential_13/dense_69/MatMul/ReadVariableOp2^
-sequential_13/dense_70/BiasAdd/ReadVariableOp-sequential_13/dense_70/BiasAdd/ReadVariableOp2\
,sequential_13/dense_70/MatMul/ReadVariableOp,sequential_13/dense_70/MatMul/ReadVariableOp2^
-sequential_13/dense_71/BiasAdd/ReadVariableOp-sequential_13/dense_71/BiasAdd/ReadVariableOp2\
,sequential_13/dense_71/MatMul/ReadVariableOp,sequential_13/dense_71/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�,
�
F__inference_dense_70_layer_call_and_return_conditional_losses_21798945

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_70/bias/Regularizer/Abs/ReadVariableOp�/dense_70/bias/Regularizer/L2Loss/ReadVariableOp�.dense_70/kernel/Regularizer/Abs/ReadVariableOp�1dense_70/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_70/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_70/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_70/kernel/Regularizer/AbsAbs6dense_70/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_70/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_70/kernel/Regularizer/SumSum#dense_70/kernel/Regularizer/Abs:y:0,dense_70/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_70/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/kernel/Regularizer/mulMul*dense_70/kernel/Regularizer/mul/x:output:0(dense_70/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/kernel/Regularizer/addAddV2*dense_70/kernel/Regularizer/Const:output:0#dense_70/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_70/kernel/Regularizer/L2LossL2Loss9dense_70/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_70/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_70/kernel/Regularizer/mul_1Mul,dense_70/kernel/Regularizer/mul_1/x:output:0+dense_70/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_70/kernel/Regularizer/add_1AddV2#dense_70/kernel/Regularizer/add:z:0%dense_70/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_70/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_70/bias/Regularizer/AbsAbs4dense_70/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_70/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_70/bias/Regularizer/SumSum!dense_70/bias/Regularizer/Abs:y:0*dense_70/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_70/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_70/bias/Regularizer/mulMul(dense_70/bias/Regularizer/mul/x:output:0&dense_70/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/addAddV2(dense_70/bias/Regularizer/Const:output:0!dense_70/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_70/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_70/bias/Regularizer/L2LossL2Loss7dense_70/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_70/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_70/bias/Regularizer/mul_1Mul*dense_70/bias/Regularizer/mul_1/x:output:0)dense_70/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_70/bias/Regularizer/add_1AddV2!dense_70/bias/Regularizer/add:z:0#dense_70/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_70/bias/Regularizer/Abs/ReadVariableOp0^dense_70/bias/Regularizer/L2Loss/ReadVariableOp/^dense_70/kernel/Regularizer/Abs/ReadVariableOp2^dense_70/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_70/bias/Regularizer/Abs/ReadVariableOp,dense_70/bias/Regularizer/Abs/ReadVariableOp2b
/dense_70/bias/Regularizer/L2Loss/ReadVariableOp/dense_70/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_70/kernel/Regularizer/Abs/ReadVariableOp.dense_70/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp1dense_70/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�,
�
F__inference_dense_66_layer_call_and_return_conditional_losses_21798761

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,dense_66/bias/Regularizer/Abs/ReadVariableOp�/dense_66/bias/Regularizer/L2Loss/ReadVariableOp�.dense_66/kernel/Regularizer/Abs/ReadVariableOp�1dense_66/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������f
!dense_66/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
.dense_66/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_66/kernel/Regularizer/AbsAbs6dense_66/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��t
#dense_66/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       �
dense_66/kernel/Regularizer/SumSum#dense_66/kernel/Regularizer/Abs:y:0,dense_66/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_66/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/kernel/Regularizer/mulMul*dense_66/kernel/Regularizer/mul/x:output:0(dense_66/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/kernel/Regularizer/addAddV2*dense_66/kernel/Regularizer/Const:output:0#dense_66/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: �
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
"dense_66/kernel/Regularizer/L2LossL2Loss9dense_66/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#dense_66/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!dense_66/kernel/Regularizer/mul_1Mul,dense_66/kernel/Regularizer/mul_1/x:output:0+dense_66/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
!dense_66/kernel/Regularizer/add_1AddV2#dense_66/kernel/Regularizer/add:z:0%dense_66/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
,dense_66/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_66/bias/Regularizer/AbsAbs4dense_66/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes	
:�k
!dense_66/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_66/bias/Regularizer/SumSum!dense_66/bias/Regularizer/Abs:y:0*dense_66/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: d
dense_66/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
dense_66/bias/Regularizer/mulMul(dense_66/bias/Regularizer/mul/x:output:0&dense_66/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/addAddV2(dense_66/bias/Regularizer/Const:output:0!dense_66/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: �
/dense_66/bias/Regularizer/L2Loss/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 dense_66/bias/Regularizer/L2LossL2Loss7dense_66/bias/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_66/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
dense_66/bias/Regularizer/mul_1Mul*dense_66/bias/Regularizer/mul_1/x:output:0)dense_66/bias/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
dense_66/bias/Regularizer/add_1AddV2!dense_66/bias/Regularizer/add:z:0#dense_66/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense_66/bias/Regularizer/Abs/ReadVariableOp0^dense_66/bias/Regularizer/L2Loss/ReadVariableOp/^dense_66/kernel/Regularizer/Abs/ReadVariableOp2^dense_66/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense_66/bias/Regularizer/Abs/ReadVariableOp,dense_66/bias/Regularizer/Abs/ReadVariableOp2b
/dense_66/bias/Regularizer/L2Loss/ReadVariableOp/dense_66/bias/Regularizer/L2Loss/ReadVariableOp2`
.dense_66/kernel/Regularizer/Abs/ReadVariableOp.dense_66/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp1dense_66/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_70_layer_call_fn_21798908

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_70_layer_call_and_return_conditional_losses_21797802p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
21798904:($
"
_user_specified_name
21798902:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
\
normalization_inputE
%serving_default_normalization_input:0������������������<
dense_710
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
�
0
1
2
!3
"4
)5
*6
17
28
99
:10
A11
B12
I13
J14
Q15
R16"
trackable_list_wrapper
�
!0
"1
)2
*3
14
25
96
:7
A8
B9
I10
J11
Q12
R13"
trackable_list_wrapper
v
S0
T1
U2
V3
W4
X5
Y6
Z7
[8
\9
]10
^11"
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_0
etrace_12�
0__inference_sequential_13_layer_call_fn_21798219
0__inference_sequential_13_layer_call_fn_21798256�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1
�
ftrace_0
gtrace_12�
K__inference_sequential_13_layer_call_and_return_conditional_losses_21797980
K__inference_sequential_13_layer_call_and_return_conditional_losses_21798182�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zftrace_0zgtrace_1
�
h	capture_0
i	capture_1B�
#__inference__wrapped_model_21797546normalization_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zh	capture_0zi	capture_1
�
j
_variables
k_iterations
l_current_learning_rate
m_index_dict
n
_momentums
o_velocities
p_update_step_xla"
experimentalOptimizer
,
qserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
�
rtrace_02�
__inference_adapt_step_1516630�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zrtrace_0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�
xtrace_02�
+__inference_dense_65_layer_call_fn_21798678�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zxtrace_0
�
ytrace_02�
F__inference_dense_65_layer_call_and_return_conditional_losses_21798715�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zytrace_0
": 	�2dense_65/kernel
:�2dense_65/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
�
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
+__inference_dense_66_layer_call_fn_21798724�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
�trace_02�
F__inference_dense_66_layer_call_and_return_conditional_losses_21798761�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_66/kernel
:�2dense_66/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_67_layer_call_fn_21798770�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_67_layer_call_and_return_conditional_losses_21798807�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_67/kernel
:�2dense_67/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_68_layer_call_fn_21798816�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_68_layer_call_and_return_conditional_losses_21798853�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_68/kernel
:�2dense_68/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_69_layer_call_fn_21798862�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_69_layer_call_and_return_conditional_losses_21798899�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_69/kernel
:�2dense_69/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_70_layer_call_fn_21798908�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_70_layer_call_and_return_conditional_losses_21798945�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_70/kernel
:�2dense_70/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_71_layer_call_fn_21798954�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_71_layer_call_and_return_conditional_losses_21798964�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�2dense_71/kernel
:2dense_71/bias
�
�trace_02�
__inference_loss_fn_0_21798981�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_21798998�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_21799015�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_21799032�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_21799049�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_21799066�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_21799083�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_7_21799100�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_8_21799117�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_9_21799134�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_10_21799151�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_11_21799168�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
5
0
1
2"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
h	capture_0
i	capture_1B�
0__inference_sequential_13_layer_call_fn_21798219normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zh	capture_0zi	capture_1
�
h	capture_0
i	capture_1B�
0__inference_sequential_13_layer_call_fn_21798256normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zh	capture_0zi	capture_1
�
h	capture_0
i	capture_1B�
K__inference_sequential_13_layer_call_and_return_conditional_losses_21797980normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zh	capture_0zi	capture_1
�
h	capture_0
i	capture_1B�
K__inference_sequential_13_layer_call_and_return_conditional_losses_21798182normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zh	capture_0zi	capture_1
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�
k0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28"
trackable_list_wrapper
:	 2	iteration
: 2current_learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
h	capture_0
i	capture_1B�
&__inference_signature_wrapper_21798513normalization_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 (

kwonlyargs�
jnormalization_input
kwonlydefaults
 
annotations� *
 zh	capture_0zi	capture_1
�B�
__inference_adapt_step_1516630iterator"�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_65_layer_call_fn_21798678inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_65_layer_call_and_return_conditional_losses_21798715inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_66_layer_call_fn_21798724inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_66_layer_call_and_return_conditional_losses_21798761inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_67_layer_call_fn_21798770inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_67_layer_call_and_return_conditional_losses_21798807inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_68_layer_call_fn_21798816inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_68_layer_call_and_return_conditional_losses_21798853inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_69_layer_call_fn_21798862inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_69_layer_call_and_return_conditional_losses_21798899inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_70_layer_call_fn_21798908inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_70_layer_call_and_return_conditional_losses_21798945inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_71_layer_call_fn_21798954inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_71_layer_call_and_return_conditional_losses_21798964inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_21798981"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_21798998"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_21799015"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_21799032"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_21799049"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_21799066"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_6_21799083"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_7_21799100"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_8_21799117"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_9_21799134"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_10_21799151"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_11_21799168"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
':%	�2Adam/m/dense_65/kernel
':%	�2Adam/v/dense_65/kernel
!:�2Adam/m/dense_65/bias
!:�2Adam/v/dense_65/bias
(:&
��2Adam/m/dense_66/kernel
(:&
��2Adam/v/dense_66/kernel
!:�2Adam/m/dense_66/bias
!:�2Adam/v/dense_66/bias
(:&
��2Adam/m/dense_67/kernel
(:&
��2Adam/v/dense_67/kernel
!:�2Adam/m/dense_67/bias
!:�2Adam/v/dense_67/bias
(:&
��2Adam/m/dense_68/kernel
(:&
��2Adam/v/dense_68/kernel
!:�2Adam/m/dense_68/bias
!:�2Adam/v/dense_68/bias
(:&
��2Adam/m/dense_69/kernel
(:&
��2Adam/v/dense_69/kernel
!:�2Adam/m/dense_69/bias
!:�2Adam/v/dense_69/bias
(:&
��2Adam/m/dense_70/kernel
(:&
��2Adam/v/dense_70/kernel
!:�2Adam/m/dense_70/bias
!:�2Adam/v/dense_70/bias
':%	�2Adam/m/dense_71/kernel
':%	�2Adam/v/dense_71/kernel
 :2Adam/m/dense_71/bias
 :2Adam/v/dense_71/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
#__inference__wrapped_model_21797546�hi!")*129:ABIJQRE�B
;�8
6�3
normalization_input������������������
� "3�0
.
dense_71"�
dense_71���������p
__inference_adapt_step_1516630NC�@
9�6
4�1�
����������IteratorSpec 
� "
 �
F__inference_dense_65_layer_call_and_return_conditional_losses_21798715d!"/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_65_layer_call_fn_21798678Y!"/�,
%�"
 �
inputs���������
� ""�
unknown�����������
F__inference_dense_66_layer_call_and_return_conditional_losses_21798761e)*0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_66_layer_call_fn_21798724Z)*0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_67_layer_call_and_return_conditional_losses_21798807e120�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_67_layer_call_fn_21798770Z120�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_68_layer_call_and_return_conditional_losses_21798853e9:0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_68_layer_call_fn_21798816Z9:0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_69_layer_call_and_return_conditional_losses_21798899eAB0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_69_layer_call_fn_21798862ZAB0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_70_layer_call_and_return_conditional_losses_21798945eIJ0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_70_layer_call_fn_21798908ZIJ0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_71_layer_call_and_return_conditional_losses_21798964dQR0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_71_layer_call_fn_21798954YQR0�-
&�#
!�
inputs����������
� "!�
unknown���������F
__inference_loss_fn_0_21798981$!�

� 
� "�
unknown G
__inference_loss_fn_10_21799151$I�

� 
� "�
unknown G
__inference_loss_fn_11_21799168$J�

� 
� "�
unknown F
__inference_loss_fn_1_21798998$"�

� 
� "�
unknown F
__inference_loss_fn_2_21799015$)�

� 
� "�
unknown F
__inference_loss_fn_3_21799032$*�

� 
� "�
unknown F
__inference_loss_fn_4_21799049$1�

� 
� "�
unknown F
__inference_loss_fn_5_21799066$2�

� 
� "�
unknown F
__inference_loss_fn_6_21799083$9�

� 
� "�
unknown F
__inference_loss_fn_7_21799100$:�

� 
� "�
unknown F
__inference_loss_fn_8_21799117$A�

� 
� "�
unknown F
__inference_loss_fn_9_21799134$B�

� 
� "�
unknown �
K__inference_sequential_13_layer_call_and_return_conditional_losses_21797980�hi!")*129:ABIJQRM�J
C�@
6�3
normalization_input������������������
p

 
� ",�)
"�
tensor_0���������
� �
K__inference_sequential_13_layer_call_and_return_conditional_losses_21798182�hi!")*129:ABIJQRM�J
C�@
6�3
normalization_input������������������
p 

 
� ",�)
"�
tensor_0���������
� �
0__inference_sequential_13_layer_call_fn_21798219�hi!")*129:ABIJQRM�J
C�@
6�3
normalization_input������������������
p

 
� "!�
unknown����������
0__inference_sequential_13_layer_call_fn_21798256�hi!")*129:ABIJQRM�J
C�@
6�3
normalization_input������������������
p 

 
� "!�
unknown����������
&__inference_signature_wrapper_21798513�hi!")*129:ABIJQR\�Y
� 
R�O
M
normalization_input6�3
normalization_input������������������"3�0
.
dense_71"�
dense_71���������