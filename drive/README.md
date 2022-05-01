注册
Post
请求参数
String：UID 昵称  必选（不需要自动生成）
string：telephone手机号，必选
string：password 密码，必选 （格式：数字+字母 大于8位小于14位  需要重复确认）
string：information_face采集人脸信息，必选  （先空出来）
string：emergency_number紧急电话号码（不能与手机号相同），必选  （最多五个）
string：name 姓名，非必选
string：gender性别，非必选
string：date_birth 出生日期，非必选
string：adress 居住地址，非必选
string: height 身高，非必选
string：weight 体重，非必选

完善个人信息
Post
请求参数
string：name 姓名，非必选
string：gender性别，非必选
string：date_birth 出生日期，非必选
string：adress 居住地址，非必选
string: height 身高，非必选
string：weight 体重，非必选

登录
Post
请求参数
string：telephone手机号，必选 
string：password密码，必选
或者 
string：face_recognition人脸识别登录 （与数据库中数据比较登录）,必选 （先空着）

添加用户管理（是不是和完善个人信息重复了）
Post
请求参数
String：UID 昵称  必选
string：telephone手机号，必选
string：password 密码，必选 （格式：数字+字母 大于8位小于14位  需要重复确认）
string：information_face采集人脸信息，必选  （先空出来）
string：emergency_number紧急电话号码（不能与手机号相同），必选  （最多五个）
string：name 姓名，非必选
string：gender性别，非必选
string：date_birth 出生日期，非必选
string：adress 居住地址，非必选
string: height 身高，非必选
string：weight 体重，非必选

查看当前用户健康报表：
Get
请求参数
Header：authorization 必选
String：begintime  endtime 起止时间 必选

查看其他用户健康报表：（不用关联其他用户，每个用户都自己注册）
Get
请求参数
Header：authorization 必选
String：用户ID 必选
String：begintime  endtime 起止时间 必选

添加车辆管理
Post
请求参数
string：CarID 车辆ID，必选
string：models车型，必选

陌生用户，车主及其家属遇到紧急情况打给提前默认的号码：（获取紧急电话号码）
Get
请求参数
Header：authorization 必选


检测到异常健康数据时发出提醒
Get
请求参数
string：在开车期间（每隔一定段时间）根据采集信息，如果超出了设定范围，返回语音提示

根据不同的用户匹配不同的健康数据
Get
请求参数
Header：authorization 必选
string：UID其他用户的ID  必须在该用户添加的其他用户列表中

//下面两个属于传感器要做的
传感器采集数据
Post
请求参数
string：heatth_data体温、心跳、血压等，非必选

摄像头识别打哈欠、闭眼等异常行为并返回给服务器，酒精传感器监测酒精浓度并返回给服务器：
Post
请求参数
string：information收集的信息，非必选  

数据库：
Userlist：用户列表
用户ID、realname(实名)、nickname（昵称）、手机号（非空）、性别、出生日期、居住地址、身高、体重

User_face_detection:人脸特征信息
先空着

Carlist：车辆信息列表
车辆ID、车辆类型、车牌（可选）

User_ emergency_contact: 紧急联系人列表
代理码、用户ID、紧急联系人手机号、和用户的关系（可选）

User_health_data: 健康数据
代理码、时间、用户ID、血压、心跳、酒精浓度……

Car_drivers: 每辆车对应的驾驶人，如果人脸识别到的驾驶人不在这个列表里面就说明是陌生人
代理码、车辆ID、用户ID

Driving_information:（传感器上传的）： 每次驾驶的信息
代理码、用户ID（default：陌生人）、开始时间、结束时间、打哈欠次数、闭眼次数、注意力不集中次数、打电话次数

User_relating_list: 用户关联的其他用户 （用户在app上查看可以他添加的其他用户的健康信息）
代理码、用户ID、其他用户ID




