import { lazy } from "react";
import { createBrowserRouter } from "react-router-dom";

// import page
const HomePage = lazy(async () => await import("@/pages/Home"));

export const MainRoutes = createBrowserRouter([
  {
    path: "/",
    element: <HomePage />,
    children: [],
  },
]);
