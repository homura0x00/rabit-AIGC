import { createApp } from "vue";
import App from "./App.vue";
// import axios from "axios"; // 跨域，前后端的中间件
import router from "./router";
import store from "./store"; // Vuex, 专门为Vue.js应用开发的"状态管理模块"
import ElementPlus from "element-plus"; // 组件库
import "element-plus/dist/index.css";


const app = createApp(App);
app.use(ElementPlus);
app.use(store);
app.use(router);

app.mount("#app");

// axios.defaults.baseURL = "https://127.0.0.1:9000/api";
// axios.defaults.xsrfCookieName = 'csrftoken';
// axios.defaults.xsrfHeaderName = 'X-CSRFToken';
// axios.defaults.withCredentials = true


