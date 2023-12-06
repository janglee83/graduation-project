import { CssBaseline, StyledEngineProvider } from "@mui/material";
import Routes from "./routes";

export default function App(): JSX.Element {
  return (
    <StyledEngineProvider injectFirst>
      <CssBaseline />
      {/* <Provider> */}
      <Routes />
      {/* </Provider> */}
    </StyledEngineProvider>
  );
}
