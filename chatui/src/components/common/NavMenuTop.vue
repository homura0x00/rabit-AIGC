<template>
  <el-menu
    router
    mode="horizontal"
    background-color="#252525"
    text-color="#fff"
    active-text-color="#ffd04b"
    style="width: 100%"
  >
    <el-menu-item route="/" index="home">LOGO</el-menu-item>
    <!-- <div class="flex-grow" /> -->
    <!-- <el-menu-item index="1">Processing Center</el-menu-item> -->
    <el-sub-menu index="person" style="position: absolute; right: 11px; padding-top: 10px;">
      <template #title>
        <el-avatar :size="40" :src="circleUrl"></el-avatar>
      </template>
      <!-- 二级菜单里的组件 -->
      <el-menu-item  route="/login" @click="login()" index="login"> 登录/注册 </el-menu-item>
    </el-sub-menu>
  </el-menu>
</template>

<script>
export default {
  name: "NavMenu",
  methods: {
    login() {
    },
  },
  data() {
    return {
      circleUrl:
        "https//cube.elemencdn.com/3/7c/3ea6beec64369c2642b92c6726f1epng.png",
    };
  },
};
</script>

<style>
.flex-grow {
  flex-grow: 1;
}
</style>
