import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, type ModelInfo, type Options, type SystemInfo } from "./lib/api";

interface AppState {
  system: SystemInfo | null;
  options: Options | null;
  models: ModelInfo[];
  refreshModels: () => Promise<void>;
  error: string | null;
}

const AppContext = createContext<AppState>({
  system: null,
  options: null,
  models: [],
  refreshModels: async () => {},
  error: null,
});

export function AppProvider({ children }: { children: ReactNode }) {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [options, setOptions] = useState<Options | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.get<ModelInfo[]>("/api/models"));
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    Promise.all([api.get<SystemInfo>("/api/system"), api.get<Options>("/api/options")])
      .then(([s, o]) => {
        setSystem(s);
        setOptions(o);
      })
      .catch((e: Error) => setError(e.message));
    refreshModels();
  }, [refreshModels]);

  return (
    <AppContext.Provider value={{ system, options, models, refreshModels, error }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp(): AppState {
  return useContext(AppContext);
}
