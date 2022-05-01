package com.example.drive.config;
import com.example.drive.entity.AuthUser;
import com.example.drive.response.RespBean;
import com.example.drive.response.ResultCode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.authentication.*;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.web.authentication.AbstractAuthenticationProcessingFilter;
import org.springframework.security.web.util.matcher.AntPathRequestMatcher;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.Date;

/**
 * 这个类负责生成token并返回给前端
 */
public class JwtLoginFilter extends AbstractAuthenticationProcessingFilter {
    private final static Logger logger= LoggerFactory.getLogger(JwtLoginFilter.class);

    protected JwtLoginFilter(String defaultFilterProcessesUrl, AuthenticationManager authenticationManager) {
        super(new AntPathRequestMatcher(defaultFilterProcessesUrl));
        logger.debug("JwtLoginFilter"+"被调用");
        setAuthenticationManager(authenticationManager);
    }
    @Override
    public Authentication attemptAuthentication(HttpServletRequest req, HttpServletResponse resp) throws AuthenticationException, IOException, ServletException, IOException {
        //这个方法是前端传json的时候用的，http请求使用getparameter
        //User user = new ObjectMapper().readValue(req.getInputStream(), User.class);
        AuthUser user = new AuthUser();
        //验证密码时最先被调用
        logger.debug("attemptAuthentication被调用");
        String username = req.getParameter("username");
        String password = req.getParameter("password");
        user.setUsername(username);
        user.setPassword(password);
        /**
         * getAuthenticationManager返回一个ProviderManager，调用ProviderManager的authenticate方法进行验证
         * authenticate会遍历所有的provider，找到处理UsernamePasswordAuthenticationToken的AbstractUserDetailsAuthenticationProvider
         * AbstractUserDetailsAuthenticationProvider的authenticate方法里首先进行用户的基本信息验证
         * 然后调用AbstractUserDetailsAuthenticationProvider的实现类DaoAuthenticationProvider的additionalAuthenticationChecks方法进行密码验证
         * 所以如果我们加入验证码或者短信验证，可以重写DaoAuthenticationProvider的additionalAuthenticationChecks方法
         * 然后判断验证码是否正确，如果正确再通过super.additionalAuthenticationChecks(userDetails, authentication)回到父类的密码验证逻辑
         * 如果自定义过滤器，将自定义的过滤器加入到 Spring Security 过滤器链中，也实现了添加登录验证码功能
         * 但是这种方式是有弊端的，就是破坏了原有的过滤器链，请求每次都要走一遍验证码过滤器，这样不合理。
         */
        return getAuthenticationManager().authenticate(new UsernamePasswordAuthenticationToken(user.getUsername(), user.getPassword()));
    }

// JWT验证成功后执行这个
    protected void successfulAuthentication(HttpServletRequest req, HttpServletResponse resp,
                                            FilterChain chain, Authentication authResult) throws IOException{
        Collection<? extends GrantedAuthority> authorities = authResult.getAuthorities();
        StringBuffer as = new StringBuffer();
        for (GrantedAuthority authority : authorities) {
            as.append(authority.getAuthority())
                    .append(",");
        }
        String jwt = Jwts.builder()
                .claim("authorities", as)//配置用户权限
                .setSubject(authResult.getName())//把用户名放到Subject，所以在JwtFilter调用方法可以获得用户名，token只包含用户名和权限
                //这里的设置时间可以通过依赖注入，现在比较麻烦稍微延长下
                .setExpiration(new Date(System.currentTimeMillis() + 3000*30*1000)) //设置过期时间
                .signWith(SignatureAlgorithm.HS512,"zhulu@123") ///用密钥加密，密钥可以随便写，但要和解密对应
                .compact();  //构建JWT
        resp.setContentType("application/json;charset=utf-8");
        PrintWriter out = resp.getWriter();
        logger.info("JWT:"+jwt);
        out.write(new ObjectMapper().writeValueAsString(RespBean.ok("登录成功",jwt)));
        out.flush();
        out.close();
    }
    //JWT验证失败执行这个
    protected void unsuccessfulAuthentication(HttpServletRequest req, HttpServletResponse resp, AuthenticationException e) throws IOException, ServletException {
        RespBean result = null;
        if (e instanceof AccountExpiredException) {
            //账号过期
            result = RespBean.error(ResultCode.USER_ACCOUNT_EXPIRED);
        } else if (e instanceof BadCredentialsException) {
            //密码错误
            result = RespBean.error(ResultCode.USER_CREDENTIALS_ERROR);
        } else if (e instanceof CredentialsExpiredException) {
            //密码过期
            result = RespBean.error(ResultCode.USER_CREDENTIALS_EXPIRED);
        } else if (e instanceof DisabledException) {
            //账号不可用
            result = RespBean.error(ResultCode.USER_ACCOUNT_DISABLE);
        } else if (e instanceof LockedException) {
            //账号锁定
            result = RespBean.error(ResultCode.USER_ACCOUNT_LOCKED);
        } else if (e instanceof InternalAuthenticationServiceException) {
            //用户不存在
            result = RespBean.error(ResultCode.USER_ACCOUNT_NOT_EXIST);
        }else{
            //其他错误
            result = RespBean.error(ResultCode.COMMON_FAIL);
        }
        resp.setContentType("application/json;charset=utf-8");
        PrintWriter out = resp.getWriter();
        out.write(new ObjectMapper().writeValueAsString(result));
        out.flush();
        out.close();
    }
}
