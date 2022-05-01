package com.example.drive.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.example.drive.entity.User;
import com.example.drive.entity.UserHealth;
import com.example.drive.mapper.UserMapper;
import com.example.drive.service.IUserService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestBody;

import java.time.LocalDateTime;
import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author zhulu
 * @since 2021-12-22
 */
@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, User> implements IUserService {
    @Autowired
    UserMapper userMapper;

    @Override
    public void register(@RequestBody User u) {
        userMapper.insert(u);
    }

    @Override
    public User perfectInformation(@RequestBody User u) {
        UpdateWrapper<User> wrapper = new UpdateWrapper<User>();
        wrapper.eq("uid",getUid());
        userMapper.update(u,wrapper);
        return u;
    }


    /**
     * 根据token获取uid
     * @return
     */
    @Override
    public Long getUid() {
        String userName =(String) SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        QueryWrapper<User> queryWrapper = new QueryWrapper<User>();
        queryWrapper.eq("tel",userName);
        User u = userMapper.selectOne(queryWrapper);
        return u.getUid();
    }

    /**
     * 根据token获取紧急联系人
     * @return
     */
    @Override
    public String getEmergencyNumber() {
        QueryWrapper<User> queryWrapper = new QueryWrapper<User>();
        queryWrapper.eq("uid",getUid());
        User u = userMapper.selectOne(queryWrapper);
        return u.getEmergencyNumber();
    }

    @Override
    public User getUser() {
        QueryWrapper<User> queryWrapper = new QueryWrapper<User>();
        queryWrapper.eq("uid",getUid());

        return userMapper.selectOne(queryWrapper);
    }
}
