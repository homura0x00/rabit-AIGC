<template>
    <div>
        <NavMenuTop2 class="nav-menu"></NavMenuTop2>
        <router-view />
    </div>
</template>

<script>
import NavMenuTop2 from '@/components/common/NavMenuTop2.vue';

export default {
    name: "HomeView",
    components: { NavMenuTop2 },
}
</script>
<style>

html, body {

    background: #f2f2f3;
    /* 设置内部填充为0，布局元素之间没有间距 */
    padding: 0px;
    /* 外部间距为0 */
    margin: 0px;
    /* 统一高度设置为100% */
    height: 100%;
}

.nav-menu {

    box-shadow: 0 2px 4px 0 rgba(0, 0, 0, .05);
}
</style>
