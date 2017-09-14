#include <Rcpp.h>
#include <cmath>
#include <vector>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(dlib)]]
#include <dlib/optimization/optimization.h>

using namespace Eigen;
using namespace std;
using namespace dlib;

typedef matrix<double,0, 1> column_vector;

struct data
{
  column_vector param;
  MatrixXd X;
  int columns;
  int rows;
};

data informations;
// [[Rcpp::export]
double fr_inv(const column_vector& param)
{
  double ans = 1;
  double xMove;
  double s1;
  double s2;
  column_vector param_temp = param;
  for(int i=informations.columns; i<informations.columns*informations.columns+informations.columns; i+=informations.columns+1)
    param_temp(i) += 0.0000002;

  for(int i = 0 ; i < informations.columns ; ++i)
  {
    s1 = 0;
    s2 = 0;

    for (int c = 0 ; c < informations.rows ; ++c)
    {
      xMove = 0;
      for (int j = 0 ; j < informations.columns ; ++j){
        xMove += param_temp(informations.columns + i * informations.columns + j) * (informations.X(c, j) - param_temp(j));
      }

      if(xMove < 0 )
        s1 += pow(xMove, 2);
      else
        s2 += pow(xMove, 2);
    }

    ans *= pow(s1, (1/3.0))+pow(s2,(1/3.0));
  }
  // determinant
  MatrixXd cov_matrix(informations.columns, informations.columns);

  for (int i=0; i< informations.columns*informations.columns; ++i)
    cov_matrix(i/informations.columns, i%informations.columns) = param_temp(i+informations.columns);

  ans *= pow( abs( cov_matrix.determinant() ), -2/3.0);

  if ((not isfinite(ans)) or (ans != ans))
    ans = numeric_limits<double>::max();

  if(ans<=0)
    return(log(0.000000000001));
  else
    return(log(ans));
}

const column_vector gr_fr(const column_vector& param)
{
  std::vector<double> ans1(informations.columns, 0);
  std::vector<double> ans_c(informations.columns * informations.columns);
  std::vector<double> s1_N(informations.columns);
  std::vector<double> s2_N(informations.columns);
  std::vector<double> s1_N_w(informations.columns);
  std::vector<double> s2_N_w(informations.columns);
  double xMove;
  double gg;
  double ggg;
  double s1, s2, g;

  column_vector ans(informations.columns+informations.columns*informations.columns);


  for(int i = 0 ; i < informations.columns ; ++i)
  {
    s1 = 0;
    s2 = 0;
    for(int c = 0 ; c<informations.columns ; ++c)
    {
      s1_N[c] = 0;
      s2_N[c] = 0;
      s1_N_w[c] = 0;
      s2_N_w[c] = 0;
    }

    for (int c = 0 ; c < informations.rows ; ++c) {
      xMove = 0;
      for (int j = 0; j < informations.columns; ++j)
        xMove += param(informations.columns + i * informations.columns + j) * (informations.X(c, j) - param(j));

      if (xMove < 0)
        s1 += pow(xMove, 2);
      else
        s2 += pow(xMove, 2);


      for (int j = 0; j < informations.columns; ++j) {
        if (xMove < 0) {
          s1_N[j] += xMove * param(informations.columns + i * informations.columns + j);
          s1_N_w[j] += 2 * xMove * (informations.X(c, j) - param(j));
        } else {
          s2_N[j] += xMove * param(informations.columns + i * informations.columns + j);
          s2_N_w[j] += 2 * xMove * (informations.X(c, j) - param(j));
        }
      }
    }

    g = pow(s1, (1/3.0))+pow(s2,(1/3.0));
    for(int c = 0 ; c < informations.columns ; ++c)
    {
      gg = 0;
      ggg = 0;
      if (s1!=0)
      {
        gg += pow(s1, -2/3.0) * s1_N[c];
        ggg += pow(s1, -2/3.0) * s1_N_w[c];
      }
      if (s2!=0)
      {
        gg += pow(s2, -2/3.0) * s2_N[c];
        ggg += pow(s2, -2/3.0) * s2_N_w[c];
      }

      gg *= (2/3.0);
      ggg *= (1/3.0);
      ans1[c] -= 1/g*gg;
      ans_c[i*informations.columns + c] = 1/g*ggg;

    }
  }

  for(int i = 0 ; i < informations.columns+informations.columns*informations.columns ; ++i)
  {
    if (i < informations.columns)
      ans(i) = ans1[i];
    else
      ans(i) = ans_c[i-informations.columns] + (-2/3.0) * param(i);
  }

  for (int i=0 ; i < informations.columns+informations.columns*informations.columns ; ++i) {
    if ((not isfinite(ans(i))) or (ans(i) != ans(i)))
      ans(i) = numeric_limits<double>::max();

  }

  return ans;
}

struct CPG_return
{
  std::vector<double> m;
  MatrixXd cov;
  std::vector<double> s;
  std::vector<double> s1;
  std::vector<double> s2;
  std::vector<double> t;
  double MLE;
  double* op_par;
  MatrixXd X;
};

CPG_return convert_param_gr()
{
  CPG_return cpg_return;
  double op_val = find_min(bfgs_search_strategy(), objective_delta_stop_strategy(1e-02), fr_inv, gr_fr, informations.param, -1);

  cpg_return.s1 = std::vector<double>(informations.columns);
  cpg_return.s2 = std::vector<double>(informations.columns);
  cpg_return.t = std::vector<double>(informations.columns);
  cpg_return.s = std::vector<double>(informations.columns);
  cpg_return.m = std::vector<double>(informations.columns);
  for (int i=0; i<informations.columns ; ++i)
    cpg_return.m[i] = informations.param(i);

  std::vector<double> gAll(informations.columns);

  double s1, s2, g, xMove;

  cpg_return.cov = Map<MatrixXd>(&informations.param(informations.columns),informations.columns, informations.columns);
  cout<<cpg_return.cov<<endl;
  cpg_return.cov = cpg_return.cov.inverse();

  for(int i = 0 ; i < informations.columns ; i++)
  {
    s1 = 0;
    s2 = 0;

    for (int c = 0 ; c < informations.rows ; ++c)
    {
      xMove = 0;
      for (int j = 0 ; j < informations.columns ; ++j)
        xMove += cpg_return.cov(i, j) * (informations.X(c, j) - cpg_return.m[j]);

      if(xMove < 0 )
        s1 += pow(xMove, 2);
      else
        s2 += pow(xMove, 2);
    }

    if (s1 < 0.000000000001)
      s1 = 0.000000000001;

    if (s2 < 0.000000000001)
      s2 = 0.000000000001;

    cpg_return.s1[i] = s1;

    cpg_return.s2[i] = s2;

    g = pow(s1, (1/3.0))+pow(s2,(1/3.0));

    gAll[i] = g;

    cpg_return.t[i] = pow((s2/s1), (1/3.0));

    cpg_return.s[i] = sqrt(pow(s1,(2/3.0))*g/informations.rows);
  }

  cpg_return.MLE = (informations.rows*informations.columns/2.0)*log((2*informations.rows)/(M_PI*exp(1))) + (-3*informations.rows/2.0)*( log( op_val ));

  for(int i = 0 ; i < informations.rows ; i++)
    for(int c = 0 ; c < informations.columns ; c++)
      informations.X(i, c)-= cpg_return.m[c];
  cpg_return.X = informations.X;

  cpg_return.op_par = new double[informations.columns+informations.columns*informations.columns];
  for (int i = 0 ; i < informations.columns+informations.columns*informations.columns ; ++i)
    cpg_return.op_par[i] = informations.param(i);

  return cpg_return;
}

// [[Rcpp::export]]
MatrixXd ica(MatrixXd X, int data_size, int data_columns)
{
  MatrixXd centered = X.rowwise() - X.colwise().mean();

  MatrixXd cov_matrix = (centered.adjoint() * centered) / double(X.rows() - 1);

  SelfAdjointEigenSolver<MatrixXd> es(cov_matrix);

  column_vector param(data_columns + data_columns*data_columns);
  for(int i = 0 ; i < data_columns ; ++i)
    param(i) = X.col(i).mean();
  for(int i = 0 ; i < data_columns*data_columns ; ++i)
    param(i + data_columns) = es.eigenvectors()(i/data_columns, i%data_columns);

  informations = {param, X, data_columns, data_size};

  CPG_return cpg_return = convert_param_gr();

  return cpg_return.X;
}

// [[Rcpp::export]]
void fun(){}
