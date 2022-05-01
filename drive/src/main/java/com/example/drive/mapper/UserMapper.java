package com.example.drive.mapper;

import com.example.drive.entity.User;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */

@Mapper
public interface UserMapper extends BaseMapper<User> {

}
