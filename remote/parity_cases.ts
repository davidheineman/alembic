import cases from './parity_cases.json';
import type { TrainConfig } from '../src/model';

export const CASES = cases as unknown as Record<string, TrainConfig>;
