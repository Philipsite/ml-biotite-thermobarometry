��
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
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
n
ConstConst*
_output_shapes

:*
dtype0*1
value(B&"2�<x�;3n�<���=�8:_�=
p
Const_1Const*
_output_shapes

:*
dtype0*1
value(B&"�X+@�d�=�?�X�?�4+<r�?
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
Adam/v/dense_58/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_58/bias/*
dtype0*
shape:*%
shared_nameAdam/v/dense_58/bias
y
(Adam/v/dense_58/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_58/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_58/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_58/bias/*
dtype0*
shape:*%
shared_nameAdam/m/dense_58/bias
y
(Adam/m/dense_58/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_58/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_58/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_58/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/v/dense_58/kernel
�
*Adam/v/dense_58/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_58/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_58/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_58/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/m/dense_58/kernel
�
*Adam/m/dense_58/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_58/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_57/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_57/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_57/bias
z
(Adam/v/dense_57/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_57/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_57/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_57/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_57/bias
z
(Adam/m/dense_57/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_57/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_57/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_57/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_57/kernel
�
*Adam/v/dense_57/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_57/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_57/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_57/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_57/kernel
�
*Adam/m/dense_57/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_57/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_56/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_56/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_56/bias
z
(Adam/v/dense_56/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_56/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_56/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_56/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_56/bias
z
(Adam/m/dense_56/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_56/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_56/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_56/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_56/kernel
�
*Adam/v/dense_56/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_56/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_56/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_56/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_56/kernel
�
*Adam/m/dense_56/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_56/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_55/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_55/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_55/bias
z
(Adam/v/dense_55/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_55/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_55/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_55/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_55/bias
z
(Adam/m/dense_55/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_55/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_55/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_55/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_55/kernel
�
*Adam/v/dense_55/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_55/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_55/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_55/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_55/kernel
�
*Adam/m/dense_55/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_55/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_54/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_54/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_54/bias
z
(Adam/v/dense_54/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_54/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_54/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_54/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_54/bias
z
(Adam/m/dense_54/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_54/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_54/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_54/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_54/kernel
�
*Adam/v/dense_54/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_54/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_54/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_54/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_54/kernel
�
*Adam/m/dense_54/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_54/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_53/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_53/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_53/bias
z
(Adam/v/dense_53/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_53/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_53/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_53/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_53/bias
z
(Adam/m/dense_53/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_53/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_53/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_53/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_53/kernel
�
*Adam/v/dense_53/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_53/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_53/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_53/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_53/kernel
�
*Adam/m/dense_53/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_53/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_52/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_52/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_52/bias
z
(Adam/v/dense_52/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_52/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_52/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_52/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_52/bias
z
(Adam/m/dense_52/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_52/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_52/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_52/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/v/dense_52/kernel
�
*Adam/v/dense_52/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_52/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_52/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_52/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/m/dense_52/kernel
�
*Adam/m/dense_52/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_52/kernel*
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
dense_58/biasVarHandleOp*
_output_shapes
: *

debug_namedense_58/bias/*
dtype0*
shape:*
shared_namedense_58/bias
k
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes
:*
dtype0
�
dense_58/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_58/kernel/*
dtype0*
shape:	�* 
shared_namedense_58/kernel
t
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel*
_output_shapes
:	�*
dtype0
�
dense_57/biasVarHandleOp*
_output_shapes
: *

debug_namedense_57/bias/*
dtype0*
shape:�*
shared_namedense_57/bias
l
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes	
:�*
dtype0
�
dense_57/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_57/kernel/*
dtype0*
shape:
��* 
shared_namedense_57/kernel
u
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel* 
_output_shapes
:
��*
dtype0
�
dense_56/biasVarHandleOp*
_output_shapes
: *

debug_namedense_56/bias/*
dtype0*
shape:�*
shared_namedense_56/bias
l
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes	
:�*
dtype0
�
dense_56/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_56/kernel/*
dtype0*
shape:
��* 
shared_namedense_56/kernel
u
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel* 
_output_shapes
:
��*
dtype0
�
dense_55/biasVarHandleOp*
_output_shapes
: *

debug_namedense_55/bias/*
dtype0*
shape:�*
shared_namedense_55/bias
l
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
_output_shapes	
:�*
dtype0
�
dense_55/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_55/kernel/*
dtype0*
shape:
��* 
shared_namedense_55/kernel
u
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel* 
_output_shapes
:
��*
dtype0
�
dense_54/biasVarHandleOp*
_output_shapes
: *

debug_namedense_54/bias/*
dtype0*
shape:�*
shared_namedense_54/bias
l
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes	
:�*
dtype0
�
dense_54/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_54/kernel/*
dtype0*
shape:
��* 
shared_namedense_54/kernel
u
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel* 
_output_shapes
:
��*
dtype0
�
dense_53/biasVarHandleOp*
_output_shapes
: *

debug_namedense_53/bias/*
dtype0*
shape:�*
shared_namedense_53/bias
l
!dense_53/bias/Read/ReadVariableOpReadVariableOpdense_53/bias*
_output_shapes	
:�*
dtype0
�
dense_53/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_53/kernel/*
dtype0*
shape:
��* 
shared_namedense_53/kernel
u
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel* 
_output_shapes
:
��*
dtype0
�
dense_52/biasVarHandleOp*
_output_shapes
: *

debug_namedense_52/bias/*
dtype0*
shape:�*
shared_namedense_52/bias
l
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes	
:�*
dtype0
�
dense_52/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_52/kernel/*
dtype0*
shape:	�* 
shared_namedense_52/kernel
t
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel*
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
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConst_1Constdense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/bias*
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
&__inference_signature_wrapper_20363330

NoOpNoOp
�|
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�|
value�|B�| B�|
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
 _adapt_function*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/_random_generator* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_random_generator* 
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
M_random_generator* 
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias*
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\_random_generator* 
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
k_random_generator* 
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias*
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
z_random_generator* 
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
0
1
2
'3
(4
65
76
E7
F8
T9
U10
c11
d12
r13
s14
�15
�16*
l
'0
(1
62
73
E4
F5
T6
U7
c8
d9
r10
s11
�12
�13*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
"
�	capture_0
�	capture_1* 
�
�
_variables
�_iterations
�_current_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 
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

�trace_0* 

'0
(1*

'0
(1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_52/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_52/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

60
71*

60
71*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_53/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_53/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

E0
F1*

E0
F1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_54/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_54/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

T0
U1*

T0
U1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_55/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_55/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

c0
d1*

c0
d1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_56/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_56/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

r0
s1*

r0
s1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_57/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_57/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_58/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_58/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*
,
�0
�1
�2
�3
�4*
* 
* 
"
�	capture_0
�	capture_1* 
"
�	capture_0
�	capture_1* 
"
�	capture_0
�	capture_1* 
"
�	capture_0
�	capture_1* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcurrent_learning_rate;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
* 
"
�	capture_0
�	capture_1* 
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
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_52/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_52/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_52/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_52/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_53/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_53/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_53/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_53/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_54/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_54/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_54/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_54/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_55/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_55/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_55/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_55/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_56/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_56/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_56/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_56/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_57/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_57/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_57/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_57/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_58/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_58/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_58/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_58/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemeanvariancecount_5dense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/bias	iterationcurrent_learning_rateAdam/m/dense_52/kernelAdam/v/dense_52/kernelAdam/m/dense_52/biasAdam/v/dense_52/biasAdam/m/dense_53/kernelAdam/v/dense_53/kernelAdam/m/dense_53/biasAdam/v/dense_53/biasAdam/m/dense_54/kernelAdam/v/dense_54/kernelAdam/m/dense_54/biasAdam/v/dense_54/biasAdam/m/dense_55/kernelAdam/v/dense_55/kernelAdam/m/dense_55/biasAdam/v/dense_55/biasAdam/m/dense_56/kernelAdam/v/dense_56/kernelAdam/m/dense_56/biasAdam/v/dense_56/biasAdam/m/dense_57/kernelAdam/v/dense_57/kernelAdam/m/dense_57/biasAdam/v/dense_57/biasAdam/m/dense_58/kernelAdam/v/dense_58/kernelAdam/m/dense_58/biasAdam/v/dense_58/biastotal_4count_4total_3count_3total_2count_2total_1count_1totalcountConst_2*F
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
!__inference__traced_save_20363997
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_5dense_52/kerneldense_52/biasdense_53/kerneldense_53/biasdense_54/kerneldense_54/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/bias	iterationcurrent_learning_rateAdam/m/dense_52/kernelAdam/v/dense_52/kernelAdam/m/dense_52/biasAdam/v/dense_52/biasAdam/m/dense_53/kernelAdam/v/dense_53/kernelAdam/m/dense_53/biasAdam/v/dense_53/biasAdam/m/dense_54/kernelAdam/v/dense_54/kernelAdam/m/dense_54/biasAdam/v/dense_54/biasAdam/m/dense_55/kernelAdam/v/dense_55/kernelAdam/m/dense_55/biasAdam/v/dense_55/biasAdam/m/dense_56/kernelAdam/v/dense_56/kernelAdam/m/dense_56/biasAdam/v/dense_56/biasAdam/m/dense_57/kernelAdam/v/dense_57/kernelAdam/m/dense_57/biasAdam/v/dense_57/biasAdam/m/dense_58/kernelAdam/v/dense_58/kernelAdam/m/dense_58/biasAdam/v/dense_58/biastotal_4count_4total_3count_3total_2count_2total_1count_1totalcount*E
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
$__inference__traced_restore_20364177Ϯ
�
H
,__inference_dropout_4_layer_call_fn_20363548

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363076a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_1_layer_call_and_return_conditional_losses_20362879

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_57_layer_call_and_return_conditional_losses_20362978

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�

f
G__inference_dropout_2_layer_call_and_return_conditional_losses_20362908

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363424

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_layer_call_and_return_conditional_losses_20362850

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_52_layer_call_and_return_conditional_losses_20362833

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
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
�
�
+__inference_dense_54_layer_call_fn_20363433

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
F__inference_dense_54_layer_call_and_return_conditional_losses_20362891p
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
20363429:($
"
_user_specified_name
20363427:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_5_layer_call_fn_20363595

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363087a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_layer_call_fn_20363355

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_20362850p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_56_layer_call_and_return_conditional_losses_20362949

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�
�
+__inference_dense_56_layer_call_fn_20363527

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
F__inference_dense_56_layer_call_and_return_conditional_losses_20362949p
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
20363523:($
"
_user_specified_name
20363521:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363565

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_layer_call_and_return_conditional_losses_20363032

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_53_layer_call_and_return_conditional_losses_20362862

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�
�
0__inference_sequential_10_layer_call_fn_20363132
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363013o
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
20363128:($
"
_user_specified_name
20363126:($
"
_user_specified_name
20363124:($
"
_user_specified_name
20363122:($
"
_user_specified_name
20363120:($
"
_user_specified_name
20363118:(
$
"
_user_specified_name
20363116:(	$
"
_user_specified_name
20363114:($
"
_user_specified_name
20363112:($
"
_user_specified_name
20363110:($
"
_user_specified_name
20363108:($
"
_user_specified_name
20363106:($
"
_user_specified_name
20363104:($
"
_user_specified_name
20363102:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
0__inference_sequential_10_layer_call_fn_20363169
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363095o
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
20363165:($
"
_user_specified_name
20363163:($
"
_user_specified_name
20363161:($
"
_user_specified_name
20363159:($
"
_user_specified_name
20363157:($
"
_user_specified_name
20363155:(
$
"
_user_specified_name
20363153:(	$
"
_user_specified_name
20363151:($
"
_user_specified_name
20363149:($
"
_user_specified_name
20363147:($
"
_user_specified_name
20363145:($
"
_user_specified_name
20363143:($
"
_user_specified_name
20363141:($
"
_user_specified_name
20363139:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
e
,__inference_dropout_4_layer_call_fn_20363543

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_20362966p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_55_layer_call_and_return_conditional_losses_20363491

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�

�
F__inference_dense_54_layer_call_and_return_conditional_losses_20362891

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�
H
,__inference_dropout_1_layer_call_fn_20363407

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363043a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�@
�
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363095
normalization_input
normalization_sub_y
normalization_sqrt_x$
dense_52_20363023:	� 
dense_52_20363025:	�%
dense_53_20363034:
�� 
dense_53_20363036:	�%
dense_54_20363045:
�� 
dense_54_20363047:	�%
dense_55_20363056:
�� 
dense_55_20363058:	�%
dense_56_20363067:
�� 
dense_56_20363069:	�%
dense_57_20363078:
�� 
dense_57_20363080:	�$
dense_58_20363089:	�
dense_58_20363091:
identity�� dense_52/StatefulPartitionedCall� dense_53/StatefulPartitionedCall� dense_54/StatefulPartitionedCall� dense_55/StatefulPartitionedCall� dense_56/StatefulPartitionedCall� dense_57/StatefulPartitionedCall� dense_58/StatefulPartitionedCallt
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
 dense_52/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_52_20363023dense_52_20363025*
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
F__inference_dense_52_layer_call_and_return_conditional_losses_20362833�
dropout/PartitionedCallPartitionedCall)dense_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_20363032�
 dense_53/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_53_20363034dense_53_20363036*
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
F__inference_dense_53_layer_call_and_return_conditional_losses_20362862�
dropout_1/PartitionedCallPartitionedCall)dense_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363043�
 dense_54/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_54_20363045dense_54_20363047*
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
F__inference_dense_54_layer_call_and_return_conditional_losses_20362891�
dropout_2/PartitionedCallPartitionedCall)dense_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363054�
 dense_55/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_55_20363056dense_55_20363058*
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
F__inference_dense_55_layer_call_and_return_conditional_losses_20362920�
dropout_3/PartitionedCallPartitionedCall)dense_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363065�
 dense_56/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_56_20363067dense_56_20363069*
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
F__inference_dense_56_layer_call_and_return_conditional_losses_20362949�
dropout_4/PartitionedCallPartitionedCall)dense_56/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363076�
 dense_57/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_57_20363078dense_57_20363080*
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
F__inference_dense_57_layer_call_and_return_conditional_losses_20362978�
dropout_5/PartitionedCallPartitionedCall)dense_57/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363087�
 dense_58/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_58_20363089dense_58_20363091*
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
F__inference_dense_58_layer_call_and_return_conditional_losses_20363006x
IdentityIdentity)dense_58/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall:($
"
_user_specified_name
20363091:($
"
_user_specified_name
20363089:($
"
_user_specified_name
20363080:($
"
_user_specified_name
20363078:($
"
_user_specified_name
20363069:($
"
_user_specified_name
20363067:(
$
"
_user_specified_name
20363058:(	$
"
_user_specified_name
20363056:($
"
_user_specified_name
20363047:($
"
_user_specified_name
20363045:($
"
_user_specified_name
20363036:($
"
_user_specified_name
20363034:($
"
_user_specified_name
20363025:($
"
_user_specified_name
20363023:$ 

_output_shapes

::$ 

_output_shapes

::e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363076

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�2
!__inference__traced_save_20363997
file_prefix)
read_disablecopyonread_mean:/
!read_1_disablecopyonread_variance:*
 read_2_disablecopyonread_count_5:	 ;
(read_3_disablecopyonread_dense_52_kernel:	�5
&read_4_disablecopyonread_dense_52_bias:	�<
(read_5_disablecopyonread_dense_53_kernel:
��5
&read_6_disablecopyonread_dense_53_bias:	�<
(read_7_disablecopyonread_dense_54_kernel:
��5
&read_8_disablecopyonread_dense_54_bias:	�<
(read_9_disablecopyonread_dense_55_kernel:
��6
'read_10_disablecopyonread_dense_55_bias:	�=
)read_11_disablecopyonread_dense_56_kernel:
��6
'read_12_disablecopyonread_dense_56_bias:	�=
)read_13_disablecopyonread_dense_57_kernel:
��6
'read_14_disablecopyonread_dense_57_bias:	�<
)read_15_disablecopyonread_dense_58_kernel:	�5
'read_16_disablecopyonread_dense_58_bias:-
#read_17_disablecopyonread_iteration:	 9
/read_18_disablecopyonread_current_learning_rate: C
0read_19_disablecopyonread_adam_m_dense_52_kernel:	�C
0read_20_disablecopyonread_adam_v_dense_52_kernel:	�=
.read_21_disablecopyonread_adam_m_dense_52_bias:	�=
.read_22_disablecopyonread_adam_v_dense_52_bias:	�D
0read_23_disablecopyonread_adam_m_dense_53_kernel:
��D
0read_24_disablecopyonread_adam_v_dense_53_kernel:
��=
.read_25_disablecopyonread_adam_m_dense_53_bias:	�=
.read_26_disablecopyonread_adam_v_dense_53_bias:	�D
0read_27_disablecopyonread_adam_m_dense_54_kernel:
��D
0read_28_disablecopyonread_adam_v_dense_54_kernel:
��=
.read_29_disablecopyonread_adam_m_dense_54_bias:	�=
.read_30_disablecopyonread_adam_v_dense_54_bias:	�D
0read_31_disablecopyonread_adam_m_dense_55_kernel:
��D
0read_32_disablecopyonread_adam_v_dense_55_kernel:
��=
.read_33_disablecopyonread_adam_m_dense_55_bias:	�=
.read_34_disablecopyonread_adam_v_dense_55_bias:	�D
0read_35_disablecopyonread_adam_m_dense_56_kernel:
��D
0read_36_disablecopyonread_adam_v_dense_56_kernel:
��=
.read_37_disablecopyonread_adam_m_dense_56_bias:	�=
.read_38_disablecopyonread_adam_v_dense_56_bias:	�D
0read_39_disablecopyonread_adam_m_dense_57_kernel:
��D
0read_40_disablecopyonread_adam_v_dense_57_kernel:
��=
.read_41_disablecopyonread_adam_m_dense_57_bias:	�=
.read_42_disablecopyonread_adam_v_dense_57_bias:	�C
0read_43_disablecopyonread_adam_m_dense_58_kernel:	�C
0read_44_disablecopyonread_adam_v_dense_58_kernel:	�<
.read_45_disablecopyonread_adam_m_dense_58_bias:<
.read_46_disablecopyonread_adam_v_dense_58_bias:+
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
Read_3/DisableCopyOnReadDisableCopyOnRead(read_3_disablecopyonread_dense_52_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp(read_3_disablecopyonread_dense_52_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
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
Read_4/DisableCopyOnReadDisableCopyOnRead&read_4_disablecopyonread_dense_52_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp&read_4_disablecopyonread_dense_52_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
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
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_dense_53_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_dense_53_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
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
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_dense_53_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_dense_53_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
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
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_54_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_54_kernel^Read_7/DisableCopyOnRead"/device:CPU:0* 
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
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_dense_54_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_dense_54_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
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
Read_9/DisableCopyOnReadDisableCopyOnRead(read_9_disablecopyonread_dense_55_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp(read_9_disablecopyonread_dense_55_kernel^Read_9/DisableCopyOnRead"/device:CPU:0* 
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
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_dense_55_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_dense_55_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
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
Read_11/DisableCopyOnReadDisableCopyOnRead)read_11_disablecopyonread_dense_56_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp)read_11_disablecopyonread_dense_56_kernel^Read_11/DisableCopyOnRead"/device:CPU:0* 
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
Read_12/DisableCopyOnReadDisableCopyOnRead'read_12_disablecopyonread_dense_56_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp'read_12_disablecopyonread_dense_56_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
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
Read_13/DisableCopyOnReadDisableCopyOnRead)read_13_disablecopyonread_dense_57_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp)read_13_disablecopyonread_dense_57_kernel^Read_13/DisableCopyOnRead"/device:CPU:0* 
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
Read_14/DisableCopyOnReadDisableCopyOnRead'read_14_disablecopyonread_dense_57_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp'read_14_disablecopyonread_dense_57_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
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
Read_15/DisableCopyOnReadDisableCopyOnRead)read_15_disablecopyonread_dense_58_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp)read_15_disablecopyonread_dense_58_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
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
Read_16/DisableCopyOnReadDisableCopyOnRead'read_16_disablecopyonread_dense_58_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp'read_16_disablecopyonread_dense_58_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
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
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_m_dense_52_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_m_dense_52_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
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
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_adam_v_dense_52_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_adam_v_dense_52_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
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
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_adam_m_dense_52_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_adam_m_dense_52_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
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
Read_22/DisableCopyOnReadDisableCopyOnRead.read_22_disablecopyonread_adam_v_dense_52_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp.read_22_disablecopyonread_adam_v_dense_52_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
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
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_m_dense_53_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_m_dense_53_kernel^Read_23/DisableCopyOnRead"/device:CPU:0* 
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
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_adam_v_dense_53_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_adam_v_dense_53_kernel^Read_24/DisableCopyOnRead"/device:CPU:0* 
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
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_m_dense_53_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_m_dense_53_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
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
Read_26/DisableCopyOnReadDisableCopyOnRead.read_26_disablecopyonread_adam_v_dense_53_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp.read_26_disablecopyonread_adam_v_dense_53_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
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
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_m_dense_54_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_m_dense_54_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
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
Read_28/DisableCopyOnReadDisableCopyOnRead0read_28_disablecopyonread_adam_v_dense_54_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp0read_28_disablecopyonread_adam_v_dense_54_kernel^Read_28/DisableCopyOnRead"/device:CPU:0* 
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
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_m_dense_54_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_m_dense_54_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
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
Read_30/DisableCopyOnReadDisableCopyOnRead.read_30_disablecopyonread_adam_v_dense_54_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp.read_30_disablecopyonread_adam_v_dense_54_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
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
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_m_dense_55_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_m_dense_55_kernel^Read_31/DisableCopyOnRead"/device:CPU:0* 
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
Read_32/DisableCopyOnReadDisableCopyOnRead0read_32_disablecopyonread_adam_v_dense_55_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp0read_32_disablecopyonread_adam_v_dense_55_kernel^Read_32/DisableCopyOnRead"/device:CPU:0* 
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
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_m_dense_55_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_m_dense_55_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
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
Read_34/DisableCopyOnReadDisableCopyOnRead.read_34_disablecopyonread_adam_v_dense_55_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp.read_34_disablecopyonread_adam_v_dense_55_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
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
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_m_dense_56_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_m_dense_56_kernel^Read_35/DisableCopyOnRead"/device:CPU:0* 
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
Read_36/DisableCopyOnReadDisableCopyOnRead0read_36_disablecopyonread_adam_v_dense_56_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp0read_36_disablecopyonread_adam_v_dense_56_kernel^Read_36/DisableCopyOnRead"/device:CPU:0* 
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
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_m_dense_56_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_m_dense_56_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
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
Read_38/DisableCopyOnReadDisableCopyOnRead.read_38_disablecopyonread_adam_v_dense_56_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp.read_38_disablecopyonread_adam_v_dense_56_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
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
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_m_dense_57_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_m_dense_57_kernel^Read_39/DisableCopyOnRead"/device:CPU:0* 
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
Read_40/DisableCopyOnReadDisableCopyOnRead0read_40_disablecopyonread_adam_v_dense_57_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp0read_40_disablecopyonread_adam_v_dense_57_kernel^Read_40/DisableCopyOnRead"/device:CPU:0* 
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
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_m_dense_57_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_m_dense_57_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
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
Read_42/DisableCopyOnReadDisableCopyOnRead.read_42_disablecopyonread_adam_v_dense_57_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp.read_42_disablecopyonread_adam_v_dense_57_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
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
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_m_dense_58_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_m_dense_58_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
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
Read_44/DisableCopyOnReadDisableCopyOnRead0read_44_disablecopyonread_adam_v_dense_58_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp0read_44_disablecopyonread_adam_v_dense_58_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
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
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_m_dense_58_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_m_dense_58_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
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
Read_46/DisableCopyOnReadDisableCopyOnRead.read_46_disablecopyonread_adam_v_dense_58_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp.read_46_disablecopyonread_adam_v_dense_58_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
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
_user_specified_nameAdam/v/dense_58/bias:4.0
.
_user_specified_nameAdam/m/dense_58/bias:6-2
0
_user_specified_nameAdam/v/dense_58/kernel:6,2
0
_user_specified_nameAdam/m/dense_58/kernel:4+0
.
_user_specified_nameAdam/v/dense_57/bias:4*0
.
_user_specified_nameAdam/m/dense_57/bias:6)2
0
_user_specified_nameAdam/v/dense_57/kernel:6(2
0
_user_specified_nameAdam/m/dense_57/kernel:4'0
.
_user_specified_nameAdam/v/dense_56/bias:4&0
.
_user_specified_nameAdam/m/dense_56/bias:6%2
0
_user_specified_nameAdam/v/dense_56/kernel:6$2
0
_user_specified_nameAdam/m/dense_56/kernel:4#0
.
_user_specified_nameAdam/v/dense_55/bias:4"0
.
_user_specified_nameAdam/m/dense_55/bias:6!2
0
_user_specified_nameAdam/v/dense_55/kernel:6 2
0
_user_specified_nameAdam/m/dense_55/kernel:40
.
_user_specified_nameAdam/v/dense_54/bias:40
.
_user_specified_nameAdam/m/dense_54/bias:62
0
_user_specified_nameAdam/v/dense_54/kernel:62
0
_user_specified_nameAdam/m/dense_54/kernel:40
.
_user_specified_nameAdam/v/dense_53/bias:40
.
_user_specified_nameAdam/m/dense_53/bias:62
0
_user_specified_nameAdam/v/dense_53/kernel:62
0
_user_specified_nameAdam/m/dense_53/kernel:40
.
_user_specified_nameAdam/v/dense_52/bias:40
.
_user_specified_nameAdam/m/dense_52/bias:62
0
_user_specified_nameAdam/v/dense_52/kernel:62
0
_user_specified_nameAdam/m/dense_52/kernel:51
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_58/bias:/+
)
_user_specified_namedense_58/kernel:-)
'
_user_specified_namedense_57/bias:/+
)
_user_specified_namedense_57/kernel:-)
'
_user_specified_namedense_56/bias:/+
)
_user_specified_namedense_56/kernel:-)
'
_user_specified_namedense_55/bias:/
+
)
_user_specified_namedense_55/kernel:-	)
'
_user_specified_namedense_54/bias:/+
)
_user_specified_namedense_54/kernel:-)
'
_user_specified_namedense_53/bias:/+
)
_user_specified_namedense_53/kernel:-)
'
_user_specified_namedense_52/bias:/+
)
_user_specified_namedense_52/kernel:'#
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

f
G__inference_dropout_4_layer_call_and_return_conditional_losses_20362966

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_53_layer_call_and_return_conditional_losses_20363397

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�

f
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363466

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
,__inference_dropout_5_layer_call_fn_20363590

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_20362995p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_5_layer_call_and_return_conditional_losses_20362995

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_52_layer_call_fn_20363339

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
F__inference_dense_52_layer_call_and_return_conditional_losses_20362833p
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
20363335:($
"
_user_specified_name
20363333:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_dropout_3_layer_call_fn_20363501

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363065a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_54_layer_call_and_return_conditional_losses_20363444

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
��
�!
$__inference__traced_restore_20364177
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:$
assignvariableop_2_count_5:	 5
"assignvariableop_3_dense_52_kernel:	�/
 assignvariableop_4_dense_52_bias:	�6
"assignvariableop_5_dense_53_kernel:
��/
 assignvariableop_6_dense_53_bias:	�6
"assignvariableop_7_dense_54_kernel:
��/
 assignvariableop_8_dense_54_bias:	�6
"assignvariableop_9_dense_55_kernel:
��0
!assignvariableop_10_dense_55_bias:	�7
#assignvariableop_11_dense_56_kernel:
��0
!assignvariableop_12_dense_56_bias:	�7
#assignvariableop_13_dense_57_kernel:
��0
!assignvariableop_14_dense_57_bias:	�6
#assignvariableop_15_dense_58_kernel:	�/
!assignvariableop_16_dense_58_bias:'
assignvariableop_17_iteration:	 3
)assignvariableop_18_current_learning_rate: =
*assignvariableop_19_adam_m_dense_52_kernel:	�=
*assignvariableop_20_adam_v_dense_52_kernel:	�7
(assignvariableop_21_adam_m_dense_52_bias:	�7
(assignvariableop_22_adam_v_dense_52_bias:	�>
*assignvariableop_23_adam_m_dense_53_kernel:
��>
*assignvariableop_24_adam_v_dense_53_kernel:
��7
(assignvariableop_25_adam_m_dense_53_bias:	�7
(assignvariableop_26_adam_v_dense_53_bias:	�>
*assignvariableop_27_adam_m_dense_54_kernel:
��>
*assignvariableop_28_adam_v_dense_54_kernel:
��7
(assignvariableop_29_adam_m_dense_54_bias:	�7
(assignvariableop_30_adam_v_dense_54_bias:	�>
*assignvariableop_31_adam_m_dense_55_kernel:
��>
*assignvariableop_32_adam_v_dense_55_kernel:
��7
(assignvariableop_33_adam_m_dense_55_bias:	�7
(assignvariableop_34_adam_v_dense_55_bias:	�>
*assignvariableop_35_adam_m_dense_56_kernel:
��>
*assignvariableop_36_adam_v_dense_56_kernel:
��7
(assignvariableop_37_adam_m_dense_56_bias:	�7
(assignvariableop_38_adam_v_dense_56_bias:	�>
*assignvariableop_39_adam_m_dense_57_kernel:
��>
*assignvariableop_40_adam_v_dense_57_kernel:
��7
(assignvariableop_41_adam_m_dense_57_bias:	�7
(assignvariableop_42_adam_v_dense_57_bias:	�=
*assignvariableop_43_adam_m_dense_58_kernel:	�=
*assignvariableop_44_adam_v_dense_58_kernel:	�6
(assignvariableop_45_adam_m_dense_58_bias:6
(assignvariableop_46_adam_v_dense_58_bias:%
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
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_52_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_52_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_53_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_53_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_54_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_54_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_55_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_55_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_56_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_56_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_57_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_57_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_58_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp!assignvariableop_16_dense_58_biasIdentity_16:output:0"/device:CPU:0*&
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
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_m_dense_52_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_v_dense_52_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_m_dense_52_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_v_dense_52_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_m_dense_53_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_v_dense_53_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_m_dense_53_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_v_dense_53_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_m_dense_54_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_v_dense_54_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_m_dense_54_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_v_dense_54_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_m_dense_55_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_v_dense_55_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_m_dense_55_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_v_dense_55_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_m_dense_56_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_v_dense_56_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_m_dense_56_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_v_dense_56_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_m_dense_57_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_v_dense_57_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_m_dense_57_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_v_dense_57_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_m_dense_58_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_v_dense_58_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_m_dense_58_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_v_dense_58_biasIdentity_46:output:0"/device:CPU:0*&
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
_user_specified_nameAdam/v/dense_58/bias:4.0
.
_user_specified_nameAdam/m/dense_58/bias:6-2
0
_user_specified_nameAdam/v/dense_58/kernel:6,2
0
_user_specified_nameAdam/m/dense_58/kernel:4+0
.
_user_specified_nameAdam/v/dense_57/bias:4*0
.
_user_specified_nameAdam/m/dense_57/bias:6)2
0
_user_specified_nameAdam/v/dense_57/kernel:6(2
0
_user_specified_nameAdam/m/dense_57/kernel:4'0
.
_user_specified_nameAdam/v/dense_56/bias:4&0
.
_user_specified_nameAdam/m/dense_56/bias:6%2
0
_user_specified_nameAdam/v/dense_56/kernel:6$2
0
_user_specified_nameAdam/m/dense_56/kernel:4#0
.
_user_specified_nameAdam/v/dense_55/bias:4"0
.
_user_specified_nameAdam/m/dense_55/bias:6!2
0
_user_specified_nameAdam/v/dense_55/kernel:6 2
0
_user_specified_nameAdam/m/dense_55/kernel:40
.
_user_specified_nameAdam/v/dense_54/bias:40
.
_user_specified_nameAdam/m/dense_54/bias:62
0
_user_specified_nameAdam/v/dense_54/kernel:62
0
_user_specified_nameAdam/m/dense_54/kernel:40
.
_user_specified_nameAdam/v/dense_53/bias:40
.
_user_specified_nameAdam/m/dense_53/bias:62
0
_user_specified_nameAdam/v/dense_53/kernel:62
0
_user_specified_nameAdam/m/dense_53/kernel:40
.
_user_specified_nameAdam/v/dense_52/bias:40
.
_user_specified_nameAdam/m/dense_52/bias:62
0
_user_specified_nameAdam/v/dense_52/kernel:62
0
_user_specified_nameAdam/m/dense_52/kernel:51
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_58/bias:/+
)
_user_specified_namedense_58/kernel:-)
'
_user_specified_namedense_57/bias:/+
)
_user_specified_namedense_57/kernel:-)
'
_user_specified_namedense_56/bias:/+
)
_user_specified_namedense_56/kernel:-)
'
_user_specified_namedense_55/bias:/
+
)
_user_specified_namedense_55/kernel:-	)
'
_user_specified_namedense_54/bias:/+
)
_user_specified_namedense_54/kernel:-)
'
_user_specified_namedense_53/bias:/+
)
_user_specified_namedense_53/kernel:-)
'
_user_specified_namedense_52/bias:/+
)
_user_specified_namedense_52/kernel:'#
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
�
�
+__inference_dense_53_layer_call_fn_20363386

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
F__inference_dense_53_layer_call_and_return_conditional_losses_20362862p
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
20363382:($
"
_user_specified_name
20363380:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363607

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363043

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363612

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_56_layer_call_and_return_conditional_losses_20363538

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�
e
,__inference_dropout_3_layer_call_fn_20363496

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_20362937p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�(
�
__inference_adapt_step_2362312
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
�

�
F__inference_dense_52_layer_call_and_return_conditional_losses_20363350

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
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
�
c
E__inference_dropout_layer_call_and_return_conditional_losses_20363377

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_3_layer_call_and_return_conditional_losses_20362937

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_58_layer_call_and_return_conditional_losses_20363006

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
�
e
,__inference_dropout_1_layer_call_fn_20363402

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_20362879p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363065

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_layer_call_fn_20363360

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_20363032a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
,__inference_dropout_2_layer_call_fn_20363449

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_20362908p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_layer_call_and_return_conditional_losses_20363372

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�_
�
#__inference__wrapped_model_20362813
normalization_input%
!sequential_10_normalization_sub_y&
"sequential_10_normalization_sqrt_xH
5sequential_10_dense_52_matmul_readvariableop_resource:	�E
6sequential_10_dense_52_biasadd_readvariableop_resource:	�I
5sequential_10_dense_53_matmul_readvariableop_resource:
��E
6sequential_10_dense_53_biasadd_readvariableop_resource:	�I
5sequential_10_dense_54_matmul_readvariableop_resource:
��E
6sequential_10_dense_54_biasadd_readvariableop_resource:	�I
5sequential_10_dense_55_matmul_readvariableop_resource:
��E
6sequential_10_dense_55_biasadd_readvariableop_resource:	�I
5sequential_10_dense_56_matmul_readvariableop_resource:
��E
6sequential_10_dense_56_biasadd_readvariableop_resource:	�I
5sequential_10_dense_57_matmul_readvariableop_resource:
��E
6sequential_10_dense_57_biasadd_readvariableop_resource:	�H
5sequential_10_dense_58_matmul_readvariableop_resource:	�D
6sequential_10_dense_58_biasadd_readvariableop_resource:
identity��-sequential_10/dense_52/BiasAdd/ReadVariableOp�,sequential_10/dense_52/MatMul/ReadVariableOp�-sequential_10/dense_53/BiasAdd/ReadVariableOp�,sequential_10/dense_53/MatMul/ReadVariableOp�-sequential_10/dense_54/BiasAdd/ReadVariableOp�,sequential_10/dense_54/MatMul/ReadVariableOp�-sequential_10/dense_55/BiasAdd/ReadVariableOp�,sequential_10/dense_55/MatMul/ReadVariableOp�-sequential_10/dense_56/BiasAdd/ReadVariableOp�,sequential_10/dense_56/MatMul/ReadVariableOp�-sequential_10/dense_57/BiasAdd/ReadVariableOp�,sequential_10/dense_57/MatMul/ReadVariableOp�-sequential_10/dense_58/BiasAdd/ReadVariableOp�,sequential_10/dense_58/MatMul/ReadVariableOp�
sequential_10/normalization/subSubnormalization_input!sequential_10_normalization_sub_y*
T0*'
_output_shapes
:���������u
 sequential_10/normalization/SqrtSqrt"sequential_10_normalization_sqrt_x*
T0*
_output_shapes

:j
%sequential_10/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
#sequential_10/normalization/MaximumMaximum$sequential_10/normalization/Sqrt:y:0.sequential_10/normalization/Maximum/y:output:0*
T0*
_output_shapes

:�
#sequential_10/normalization/truedivRealDiv#sequential_10/normalization/sub:z:0'sequential_10/normalization/Maximum:z:0*
T0*'
_output_shapes
:����������
,sequential_10/dense_52/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_52_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_10/dense_52/MatMulMatMul'sequential_10/normalization/truediv:z:04sequential_10/dense_52/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_10/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_52_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_10/dense_52/BiasAddBiasAdd'sequential_10/dense_52/MatMul:product:05sequential_10/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_10/dense_52/ReluRelu'sequential_10/dense_52/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
sequential_10/dropout/IdentityIdentity)sequential_10/dense_52/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_10/dense_53/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_53_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_10/dense_53/MatMulMatMul'sequential_10/dropout/Identity:output:04sequential_10/dense_53/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_10/dense_53/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_53_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_10/dense_53/BiasAddBiasAdd'sequential_10/dense_53/MatMul:product:05sequential_10/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_10/dense_53/ReluRelu'sequential_10/dense_53/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 sequential_10/dropout_1/IdentityIdentity)sequential_10/dense_53/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_10/dense_54/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_54_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_10/dense_54/MatMulMatMul)sequential_10/dropout_1/Identity:output:04sequential_10/dense_54/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_10/dense_54/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_54_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_10/dense_54/BiasAddBiasAdd'sequential_10/dense_54/MatMul:product:05sequential_10/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_10/dense_54/ReluRelu'sequential_10/dense_54/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 sequential_10/dropout_2/IdentityIdentity)sequential_10/dense_54/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_10/dense_55/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_55_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_10/dense_55/MatMulMatMul)sequential_10/dropout_2/Identity:output:04sequential_10/dense_55/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_10/dense_55/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_55_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_10/dense_55/BiasAddBiasAdd'sequential_10/dense_55/MatMul:product:05sequential_10/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_10/dense_55/ReluRelu'sequential_10/dense_55/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 sequential_10/dropout_3/IdentityIdentity)sequential_10/dense_55/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_10/dense_56/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_56_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_10/dense_56/MatMulMatMul)sequential_10/dropout_3/Identity:output:04sequential_10/dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_10/dense_56/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_56_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_10/dense_56/BiasAddBiasAdd'sequential_10/dense_56/MatMul:product:05sequential_10/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_10/dense_56/ReluRelu'sequential_10/dense_56/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 sequential_10/dropout_4/IdentityIdentity)sequential_10/dense_56/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_10/dense_57/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_57_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_10/dense_57/MatMulMatMul)sequential_10/dropout_4/Identity:output:04sequential_10/dense_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_10/dense_57/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_57_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_10/dense_57/BiasAddBiasAdd'sequential_10/dense_57/MatMul:product:05sequential_10/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_10/dense_57/ReluRelu'sequential_10/dense_57/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 sequential_10/dropout_5/IdentityIdentity)sequential_10/dense_57/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_10/dense_58/MatMul/ReadVariableOpReadVariableOp5sequential_10_dense_58_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_10/dense_58/MatMulMatMul)sequential_10/dropout_5/Identity:output:04sequential_10/dense_58/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_10/dense_58/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_10/dense_58/BiasAddBiasAdd'sequential_10/dense_58/MatMul:product:05sequential_10/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'sequential_10/dense_58/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_10/dense_52/BiasAdd/ReadVariableOp-^sequential_10/dense_52/MatMul/ReadVariableOp.^sequential_10/dense_53/BiasAdd/ReadVariableOp-^sequential_10/dense_53/MatMul/ReadVariableOp.^sequential_10/dense_54/BiasAdd/ReadVariableOp-^sequential_10/dense_54/MatMul/ReadVariableOp.^sequential_10/dense_55/BiasAdd/ReadVariableOp-^sequential_10/dense_55/MatMul/ReadVariableOp.^sequential_10/dense_56/BiasAdd/ReadVariableOp-^sequential_10/dense_56/MatMul/ReadVariableOp.^sequential_10/dense_57/BiasAdd/ReadVariableOp-^sequential_10/dense_57/MatMul/ReadVariableOp.^sequential_10/dense_58/BiasAdd/ReadVariableOp-^sequential_10/dense_58/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 2^
-sequential_10/dense_52/BiasAdd/ReadVariableOp-sequential_10/dense_52/BiasAdd/ReadVariableOp2\
,sequential_10/dense_52/MatMul/ReadVariableOp,sequential_10/dense_52/MatMul/ReadVariableOp2^
-sequential_10/dense_53/BiasAdd/ReadVariableOp-sequential_10/dense_53/BiasAdd/ReadVariableOp2\
,sequential_10/dense_53/MatMul/ReadVariableOp,sequential_10/dense_53/MatMul/ReadVariableOp2^
-sequential_10/dense_54/BiasAdd/ReadVariableOp-sequential_10/dense_54/BiasAdd/ReadVariableOp2\
,sequential_10/dense_54/MatMul/ReadVariableOp,sequential_10/dense_54/MatMul/ReadVariableOp2^
-sequential_10/dense_55/BiasAdd/ReadVariableOp-sequential_10/dense_55/BiasAdd/ReadVariableOp2\
,sequential_10/dense_55/MatMul/ReadVariableOp,sequential_10/dense_55/MatMul/ReadVariableOp2^
-sequential_10/dense_56/BiasAdd/ReadVariableOp-sequential_10/dense_56/BiasAdd/ReadVariableOp2\
,sequential_10/dense_56/MatMul/ReadVariableOp,sequential_10/dense_56/MatMul/ReadVariableOp2^
-sequential_10/dense_57/BiasAdd/ReadVariableOp-sequential_10/dense_57/BiasAdd/ReadVariableOp2\
,sequential_10/dense_57/MatMul/ReadVariableOp,sequential_10/dense_57/MatMul/ReadVariableOp2^
-sequential_10/dense_58/BiasAdd/ReadVariableOp-sequential_10/dense_58/BiasAdd/ReadVariableOp2\
,sequential_10/dense_58/MatMul/ReadVariableOp,sequential_10/dense_58/MatMul/ReadVariableOp:($
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
�
�
+__inference_dense_55_layer_call_fn_20363480

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
F__inference_dense_55_layer_call_and_return_conditional_losses_20362920p
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
20363476:($
"
_user_specified_name
20363474:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363419

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_58_layer_call_fn_20363621

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
F__inference_dense_58_layer_call_and_return_conditional_losses_20363006o
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
20363617:($
"
_user_specified_name
20363615:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_dense_58_layer_call_and_return_conditional_losses_20363631

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
�
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363471

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_57_layer_call_fn_20363574

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
F__inference_dense_57_layer_call_and_return_conditional_losses_20362978p
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
20363570:($
"
_user_specified_name
20363568:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_dropout_2_layer_call_fn_20363454

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363054a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

f
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363513

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363087

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_55_layer_call_and_return_conditional_losses_20362920

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�
e
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363054

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_20363330
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
#__inference__wrapped_model_20362813o
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
20363326:($
"
_user_specified_name
20363324:($
"
_user_specified_name
20363322:($
"
_user_specified_name
20363320:($
"
_user_specified_name
20363318:($
"
_user_specified_name
20363316:(
$
"
_user_specified_name
20363314:(	$
"
_user_specified_name
20363312:($
"
_user_specified_name
20363310:($
"
_user_specified_name
20363308:($
"
_user_specified_name
20363306:($
"
_user_specified_name
20363304:($
"
_user_specified_name
20363302:($
"
_user_specified_name
20363300:$ 

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
F__inference_dense_57_layer_call_and_return_conditional_losses_20363585

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
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
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
�
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363518

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�I
�
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363013
normalization_input
normalization_sub_y
normalization_sqrt_x$
dense_52_20362834:	� 
dense_52_20362836:	�%
dense_53_20362863:
�� 
dense_53_20362865:	�%
dense_54_20362892:
�� 
dense_54_20362894:	�%
dense_55_20362921:
�� 
dense_55_20362923:	�%
dense_56_20362950:
�� 
dense_56_20362952:	�%
dense_57_20362979:
�� 
dense_57_20362981:	�$
dense_58_20363007:	�
dense_58_20363009:
identity�� dense_52/StatefulPartitionedCall� dense_53/StatefulPartitionedCall� dense_54/StatefulPartitionedCall� dense_55/StatefulPartitionedCall� dense_56/StatefulPartitionedCall� dense_57/StatefulPartitionedCall� dense_58/StatefulPartitionedCall�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!dropout_3/StatefulPartitionedCall�!dropout_4/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCallt
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
 dense_52/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_52_20362834dense_52_20362836*
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
F__inference_dense_52_layer_call_and_return_conditional_losses_20362833�
dropout/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_20362850�
 dense_53/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_53_20362863dense_53_20362865*
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
F__inference_dense_53_layer_call_and_return_conditional_losses_20362862�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)dense_53/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_20362879�
 dense_54/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_54_20362892dense_54_20362894*
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
F__inference_dense_54_layer_call_and_return_conditional_losses_20362891�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_2_layer_call_and_return_conditional_losses_20362908�
 dense_55/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_55_20362921dense_55_20362923*
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
F__inference_dense_55_layer_call_and_return_conditional_losses_20362920�
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_20362937�
 dense_56/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_56_20362950dense_56_20362952*
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
F__inference_dense_56_layer_call_and_return_conditional_losses_20362949�
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_20362966�
 dense_57/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_57_20362979dense_57_20362981*
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
F__inference_dense_57_layer_call_and_return_conditional_losses_20362978�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_20362995�
 dense_58/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_58_20363007dense_58_20363009*
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
F__inference_dense_58_layer_call_and_return_conditional_losses_20363006x
IdentityIdentity)dense_58/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall!^dense_57/StatefulPartitionedCall!^dense_58/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:������������������::: : : : : : : : : : : : : : 2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:($
"
_user_specified_name
20363009:($
"
_user_specified_name
20363007:($
"
_user_specified_name
20362981:($
"
_user_specified_name
20362979:($
"
_user_specified_name
20362952:($
"
_user_specified_name
20362950:(
$
"
_user_specified_name
20362923:(	$
"
_user_specified_name
20362921:($
"
_user_specified_name
20362894:($
"
_user_specified_name
20362892:($
"
_user_specified_name
20362865:($
"
_user_specified_name
20362863:($
"
_user_specified_name
20362836:($
"
_user_specified_name
20362834:$ 

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

f
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363560

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
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
dense_580
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
 _adapt_function"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/_random_generator"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>_random_generator"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
M_random_generator"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

Tkernel
Ubias"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\_random_generator"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses
k_random_generator"
_tf_keras_layer
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias"
_tf_keras_layer
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
z_random_generator"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
0
1
2
'3
(4
65
76
E7
F8
T9
U10
c11
d12
r13
s14
�15
�16"
trackable_list_wrapper
�
'0
(1
62
73
E4
F5
T6
U7
c8
d9
r10
s11
�12
�13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_sequential_10_layer_call_fn_20363132
0__inference_sequential_10_layer_call_fn_20363169�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363013
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363095�
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
 z�trace_0z�trace_1
�
�	capture_0
�	capture_1B�
#__inference__wrapped_model_20362813normalization_input"�
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
 z�	capture_0z�	capture_1
�
�
_variables
�_iterations
�_current_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
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
�trace_02�
__inference_adapt_step_2362312�
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
 z�trace_0
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_52_layer_call_fn_20363339�
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
F__inference_dense_52_layer_call_and_return_conditional_losses_20363350�
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
": 	�2dense_52/kernel
:�2dense_52/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_layer_call_fn_20363355
*__inference_dropout_layer_call_fn_20363360�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_layer_call_and_return_conditional_losses_20363372
E__inference_dropout_layer_call_and_return_conditional_losses_20363377�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_53_layer_call_fn_20363386�
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
F__inference_dense_53_layer_call_and_return_conditional_losses_20363397�
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
��2dense_53/kernel
:�2dense_53/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_1_layer_call_fn_20363402
,__inference_dropout_1_layer_call_fn_20363407�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363419
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363424�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_54_layer_call_fn_20363433�
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
F__inference_dense_54_layer_call_and_return_conditional_losses_20363444�
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
��2dense_54/kernel
:�2dense_54/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_2_layer_call_fn_20363449
,__inference_dropout_2_layer_call_fn_20363454�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363466
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363471�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_55_layer_call_fn_20363480�
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
F__inference_dense_55_layer_call_and_return_conditional_losses_20363491�
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
��2dense_55/kernel
:�2dense_55/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_3_layer_call_fn_20363496
,__inference_dropout_3_layer_call_fn_20363501�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363513
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363518�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_56_layer_call_fn_20363527�
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
F__inference_dense_56_layer_call_and_return_conditional_losses_20363538�
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
��2dense_56/kernel
:�2dense_56/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_4_layer_call_fn_20363543
,__inference_dropout_4_layer_call_fn_20363548�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363560
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363565�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_57_layer_call_fn_20363574�
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
F__inference_dense_57_layer_call_and_return_conditional_losses_20363585�
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
��2dense_57/kernel
:�2dense_57/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_dropout_5_layer_call_fn_20363590
,__inference_dropout_5_layer_call_fn_20363595�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363607
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363612�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_58_layer_call_fn_20363621�
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
F__inference_dense_58_layer_call_and_return_conditional_losses_20363631�
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
": 	�2dense_58/kernel
:2dense_58/bias
5
0
1
2"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�	capture_0
�	capture_1B�
0__inference_sequential_10_layer_call_fn_20363132normalization_input"�
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
 z�	capture_0z�	capture_1
�
�	capture_0
�	capture_1B�
0__inference_sequential_10_layer_call_fn_20363169normalization_input"�
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
 z�	capture_0z�	capture_1
�
�	capture_0
�	capture_1B�
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363013normalization_input"�
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
 z�	capture_0z�	capture_1
�
�	capture_0
�	capture_1B�
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363095normalization_input"�
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
 z�	capture_0z�	capture_1
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28"
trackable_list_wrapper
:	 2	iteration
: 2current_learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
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
�	capture_0
�	capture_1B�
&__inference_signature_wrapper_20363330normalization_input"�
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
 z�	capture_0z�	capture_1
�B�
__inference_adapt_step_2362312iterator"�
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_52_layer_call_fn_20363339inputs"�
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
F__inference_dense_52_layer_call_and_return_conditional_losses_20363350inputs"�
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
*__inference_dropout_layer_call_fn_20363355inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
*__inference_dropout_layer_call_fn_20363360inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
E__inference_dropout_layer_call_and_return_conditional_losses_20363372inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
E__inference_dropout_layer_call_and_return_conditional_losses_20363377inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
+__inference_dense_53_layer_call_fn_20363386inputs"�
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
F__inference_dense_53_layer_call_and_return_conditional_losses_20363397inputs"�
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
,__inference_dropout_1_layer_call_fn_20363402inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
,__inference_dropout_1_layer_call_fn_20363407inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363419inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363424inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
+__inference_dense_54_layer_call_fn_20363433inputs"�
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
F__inference_dense_54_layer_call_and_return_conditional_losses_20363444inputs"�
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
,__inference_dropout_2_layer_call_fn_20363449inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
,__inference_dropout_2_layer_call_fn_20363454inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363466inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363471inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
+__inference_dense_55_layer_call_fn_20363480inputs"�
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
F__inference_dense_55_layer_call_and_return_conditional_losses_20363491inputs"�
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
,__inference_dropout_3_layer_call_fn_20363496inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
,__inference_dropout_3_layer_call_fn_20363501inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363513inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363518inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
+__inference_dense_56_layer_call_fn_20363527inputs"�
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
F__inference_dense_56_layer_call_and_return_conditional_losses_20363538inputs"�
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
,__inference_dropout_4_layer_call_fn_20363543inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
,__inference_dropout_4_layer_call_fn_20363548inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363560inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363565inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
+__inference_dense_57_layer_call_fn_20363574inputs"�
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
F__inference_dense_57_layer_call_and_return_conditional_losses_20363585inputs"�
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
,__inference_dropout_5_layer_call_fn_20363590inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
,__inference_dropout_5_layer_call_fn_20363595inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363607inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363612inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
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
+__inference_dense_58_layer_call_fn_20363621inputs"�
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
F__inference_dense_58_layer_call_and_return_conditional_losses_20363631inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
':%	�2Adam/m/dense_52/kernel
':%	�2Adam/v/dense_52/kernel
!:�2Adam/m/dense_52/bias
!:�2Adam/v/dense_52/bias
(:&
��2Adam/m/dense_53/kernel
(:&
��2Adam/v/dense_53/kernel
!:�2Adam/m/dense_53/bias
!:�2Adam/v/dense_53/bias
(:&
��2Adam/m/dense_54/kernel
(:&
��2Adam/v/dense_54/kernel
!:�2Adam/m/dense_54/bias
!:�2Adam/v/dense_54/bias
(:&
��2Adam/m/dense_55/kernel
(:&
��2Adam/v/dense_55/kernel
!:�2Adam/m/dense_55/bias
!:�2Adam/v/dense_55/bias
(:&
��2Adam/m/dense_56/kernel
(:&
��2Adam/v/dense_56/kernel
!:�2Adam/m/dense_56/bias
!:�2Adam/v/dense_56/bias
(:&
��2Adam/m/dense_57/kernel
(:&
��2Adam/v/dense_57/kernel
!:�2Adam/m/dense_57/bias
!:�2Adam/v/dense_57/bias
':%	�2Adam/m/dense_58/kernel
':%	�2Adam/v/dense_58/kernel
 :2Adam/m/dense_58/bias
 :2Adam/v/dense_58/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
#__inference__wrapped_model_20362813���'(67EFTUcdrs��E�B
;�8
6�3
normalization_input������������������
� "3�0
.
dense_58"�
dense_58���������p
__inference_adapt_step_2362312NC�@
9�6
4�1�
����������IteratorSpec 
� "
 �
F__inference_dense_52_layer_call_and_return_conditional_losses_20363350d'(/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_52_layer_call_fn_20363339Y'(/�,
%�"
 �
inputs���������
� ""�
unknown�����������
F__inference_dense_53_layer_call_and_return_conditional_losses_20363397e670�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_53_layer_call_fn_20363386Z670�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_54_layer_call_and_return_conditional_losses_20363444eEF0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_54_layer_call_fn_20363433ZEF0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_55_layer_call_and_return_conditional_losses_20363491eTU0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_55_layer_call_fn_20363480ZTU0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_56_layer_call_and_return_conditional_losses_20363538ecd0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_56_layer_call_fn_20363527Zcd0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_57_layer_call_and_return_conditional_losses_20363585ers0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
+__inference_dense_57_layer_call_fn_20363574Zrs0�-
&�#
!�
inputs����������
� ""�
unknown�����������
F__inference_dense_58_layer_call_and_return_conditional_losses_20363631f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_58_layer_call_fn_20363621[��0�-
&�#
!�
inputs����������
� "!�
unknown����������
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363419e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
G__inference_dropout_1_layer_call_and_return_conditional_losses_20363424e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
,__inference_dropout_1_layer_call_fn_20363402Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
,__inference_dropout_1_layer_call_fn_20363407Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363466e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
G__inference_dropout_2_layer_call_and_return_conditional_losses_20363471e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
,__inference_dropout_2_layer_call_fn_20363449Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
,__inference_dropout_2_layer_call_fn_20363454Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363513e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
G__inference_dropout_3_layer_call_and_return_conditional_losses_20363518e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
,__inference_dropout_3_layer_call_fn_20363496Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
,__inference_dropout_3_layer_call_fn_20363501Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363560e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
G__inference_dropout_4_layer_call_and_return_conditional_losses_20363565e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
,__inference_dropout_4_layer_call_fn_20363543Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
,__inference_dropout_4_layer_call_fn_20363548Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363607e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
G__inference_dropout_5_layer_call_and_return_conditional_losses_20363612e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
,__inference_dropout_5_layer_call_fn_20363590Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
,__inference_dropout_5_layer_call_fn_20363595Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
E__inference_dropout_layer_call_and_return_conditional_losses_20363372e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
E__inference_dropout_layer_call_and_return_conditional_losses_20363377e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
*__inference_dropout_layer_call_fn_20363355Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
*__inference_dropout_layer_call_fn_20363360Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363013���'(67EFTUcdrs��M�J
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_20363095���'(67EFTUcdrs��M�J
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
0__inference_sequential_10_layer_call_fn_20363132���'(67EFTUcdrs��M�J
C�@
6�3
normalization_input������������������
p

 
� "!�
unknown����������
0__inference_sequential_10_layer_call_fn_20363169���'(67EFTUcdrs��M�J
C�@
6�3
normalization_input������������������
p 

 
� "!�
unknown����������
&__inference_signature_wrapper_20363330���'(67EFTUcdrs��\�Y
� 
R�O
M
normalization_input6�3
normalization_input������������������"3�0
.
dense_58"�
dense_58���������