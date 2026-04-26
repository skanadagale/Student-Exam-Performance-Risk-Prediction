import LogisticRegression from 'ml-logistic-regression';
import { Matrix } from 'ml-matrix';
import { ModelMetrics } from '../types';

export class StudentRiskModel {
  private model: LogisticRegression | null = null;
  private featureMean: number[] = [];
  private featureStd: number[] = [];

  constructor() {}

  private standardize(features: number[][], training: boolean = false) {
    const matrix = new Matrix(features);
    const cols = matrix.columns;
    
    if (training) {
      this.featureMean = [];
      this.featureStd = [];
      for (let i = 0; i < cols; i++) {
        const col = matrix.getColumn(i);
        const mean = col.reduce((a, b) => a + b, 0) / col.length;
        const std = Math.sqrt(col.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / col.length) || 1;
        this.featureMean.push(mean);
        this.featureStd.push(std);
      }
    }

    const standardized = features.map(row => 
      row.map((val, i) => (val - this.featureMean[i]) / this.featureStd[i])
    );
    
    return standardized;
  }

  train(trainData: { features: number[], label: number }[]) {
    const X = trainData.map(d => d.features);
    const y = trainData.map(d => d.label);

    const standardizedX = this.standardize(X, true);

    this.model = new LogisticRegression({
      numSteps: 1000,
      learningRate: 5e-2
    });

    this.model.train(new Matrix(standardizedX), Matrix.columnVector(y));
  }

  predict(features: number[]) {
    if (!this.model) throw new Error("Model not trained");
    
    const standardized = features.map((val, i) => (val - this.featureMean[i]) / this.featureStd[i]);
    const prob = this.model.predict(new Matrix([standardized]))[0];
    return {
      prediction: prob > 0.5 ? 1 : 0,
      probability: prob
    };
  }

  evaluate(testData: { features: number[], label: number }[]): ModelMetrics {
    if (!this.model) throw new Error("Model not trained");

    const X = testData.map(d => d.features);
    const yTrue = testData.map(d => d.label);
    
    const standardizedX = this.standardize(X);
    const probs = this.model.predict(new Matrix(standardizedX));
    const yPred = probs.map(p => (p > 0.5 ? 1 : 0));

    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] === 1 && yPred[i] === 1) tp++;
      else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
      else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
      else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
    }

    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = (2 * precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / yTrue.length;

    return {
      accuracy,
      precision,
      recall,
      f1,
      confusionMatrix: [[tn, fp], [fn, tp]]
    };
  }
}
