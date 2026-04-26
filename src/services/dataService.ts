import Papa from 'papaparse';
import { StudentData, ProcessedStudent, DATASET_URL } from '../types';

export async function fetchAndProcessData(): Promise<ProcessedStudent[]> {
  return new Promise((resolve, reject) => {
    Papa.parse(DATASET_URL, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      delimiter: ";", // UCI dataset uses semicolon
      complete: (results) => {
        const processed = (results.data as StudentData[]).map(student => {
          const g1 = student.G1 || 0;
          const g2 = student.G2 || 0;
          const g3 = student.G3 || 0;
          
          return {
            ...student,
            average_previous_score: (g1 + g2) / 2,
            high_risk_profile: (student.absences > 10 && student.studytime < 2) ? 1 : 0,
            is_at_risk: g3 < 10 ? 1 : 0
          };
        });
        resolve(processed);
      },
      error: (error) => reject(error)
    });
  });
}

export function encodeData(data: ProcessedStudent[]) {
  // We need to convert categorical variables to numbers for the model
  // Features requested: studytime, absences, failures, G1, G2, schoolsup, famsup, paid, internet, Dalc, Walc, romantic
  
  return data.map(s => {
    const features = [
      s.studytime,
      s.absences,
      s.failures,
      s.G1,
      s.G2,
      s.average_previous_score,
      s.schoolsup === 'yes' ? 1 : 0,
      s.famsup === 'yes' ? 1 : 0,
      s.paid === 'yes' ? 1 : 0,
      s.internet === 'yes' ? 1 : 0,
      s.Dalc,
      s.Walc,
      s.romantic === 'yes' ? 1 : 0,
      s.studytime * s.absences // Interaction feature mentioned in prompt
    ];
    
    return {
      features,
      label: s.is_at_risk
    };
  });
}

export function splitData(encoded: { features: number[], label: number }[], trainRatio: number = 0.8) {
  // Shuffle data
  const shuffled = [...encoded].sort(() => Math.random() - 0.5);
  const splitIdx = Math.floor(shuffled.length * trainRatio);
  
  return {
    train: shuffled.slice(0, splitIdx),
    test: shuffled.slice(splitIdx)
  };
}
