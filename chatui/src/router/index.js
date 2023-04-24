import { createRouter, createWebHashHistory } from "vue-router";
import HomeView from "../views/HomeView.vue";
import HomeView2 from "../views/HomeView2.vue";
import KGQACard from "@/components/KGQA/KGQACard";
import AnswerCard from "@/components/KGQA/AnswerCard";
import KnowledgeGraph from "@/components/KGQA/KnowledgeGraph";
import UserCard from "@/components/UserCenter/UserCard";

const routes = [
  {
    path: "/",
    name: "home",
    component: HomeView
  },

  {
    path: "/KGQA",
    name: "kgqahome",
    component: HomeView2,
    children: [
      {
        path: "/KGQA",
        name: "KGQA_home",
        component: KGQACard,
        meta: {
          requireAuth: false,
        }
      }
    ]
  },
  {
    path: "/KGQA",
    name: "knowlegdeGraph",
    component: HomeView2,
    children: [
      {
        path: "/KGQA/kg",
        name: "KnowledgeGraph",
        component: KnowledgeGraph,
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
        path: "/QA",
        name: "AnswerCard",
        component: AnswerCard,
        meta: {
          requireAuth: false,
        }
      }
    ]
  },

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
