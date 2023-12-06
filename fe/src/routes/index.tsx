import { RouterProvider } from "react-router-dom";
import { MainRoutes } from "./MainRoutes";

export default function Routes(): JSX.Element {
  return <RouterProvider router={MainRoutes} />;
}
