package com.example.drive.service.impl;

import com.example.drive.entity.Title;
import com.example.drive.mapper.TitleMapper;
import com.example.drive.service.ITitleService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author zhulu
 * @since 2022-02-23
 */
@Service
public class TitleServiceImpl extends ServiceImpl<TitleMapper, Title> implements ITitleService {

    @Autowired
    TitleMapper titleMapper;

    /**
     * 根据titleid获取title
     * @param id
     * @return
     */
    @Override
    public Title getTitleById(int id) {
        return titleMapper.getTitleById(id);
    }
}
