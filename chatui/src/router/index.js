import { createRouter, createWebHashHistory } from "vue-router";
import HomeView from "../views/HomeView.vue";
import HomeView2 from "../views/HomeView2.vue";
import KGCard from "@/components/KGQA/KGCard";
import ChatCard from "@/components/KGQA/ChatCard";
import UserCard from "@/components/UserCenter/UserCard";
import LoginView from "@/views/LoginView";

const routes = [
  // 网站主页
  {
    path: "/",
    name: "home",
    component: HomeView,
    children: [
      {
        path: "/login",
        name: 'Login',
        component: LoginView,
        meta: {
          requireAuth: false,
        }
      }
    ]
  },

// 知识图谱问答系统模块
  {
    path: "/KGQA",
    name: "knowlegdeGraph",
    component: HomeView2,
    children: [
      {
        path: "/KGQA/kg",
        name: "KnowledgeGraph",
        component: KGCard,
        meta: {
          requireAuth: false,
        }
      }
    ]
  },
  {
    path: "/KGQA",
    name: "AnswerCard",
    component: HomeView2,
    children: [
      {
        path: "/KGQA/Chat",
        name: "Chat",
        component: ChatCard,
        meta: {
          requireAuth: false,
        }
      }
    ]
  },
// 用户中心
  {
    path: "/KGQA/Personal",
    name: "UserCard",
    component: UserCard,
  },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;

// router.beforeEach((to, from, next)=>{
//   const isLogin = localStorage.getItem('is_login') == 'true';
//   if( isLogin ){
//     if( to.path !== '/' )
//       next();
//     else
//       next('/');
//   } else {
//     if( to.path !== '/' )
//       next('/login');
//     else
//       next();
//   }
// })