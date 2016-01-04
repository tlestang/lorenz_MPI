#include<iostream>
#include<sstream>
#include<cstdlib>
#include<fstream>

#include<cmath>
#include<mpi.h>

#define MASTER 0

using namespace std;

struct Point{
  double x;
  double y;
  double z;
};
/*Parameters of the Lorenz-63 model*/
double sigma = 10.0;
double rho = 28.0;
double beta = 8./3.;

int Nc = 10; // Number of clones
double T = 40; // Total simulation time
double dt = 0.002; // Model timestep
double dT = 40; // Cloning timestep

double eps = 0.001;

double lorenz(Point, Point*, double);
double randNormal(const double, const double);
void lorenz_transient(Point*, double);

int main()
{
  int nbrTimeSteps = floor(T/dT);  
  Point x0[Nc]; // State of the clones after cloning (initial conditions for next block)
  Point x[Nc]; // State of the clones after time evolution
  Point temp[2*Nc];
  double s[Nc]; //Absolute weight
  int nbrCreatedCopies[Nc];
  double R_record[nbrTimeSteps]; double R, total_R;

  double phi_alpha, phi_theor;
  double alpha = -0.1, alphaIncr = 1.6;
  int NcPrime, deltaN, copyIdx, k;
  int l= 0; int idx;
  int my_rank, p, local_Nc, cloneIdxMin, tag = 0;
  MPI_Status status;

  ofstream result;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&p);

  local_Nc = Nc/p;
  cloneIdxMin = my_rank*local_Nc;

  //   cout << "I'm process " << my_rank << " and I take care of clones " << cloneIdxMin << " to " << cloneIdxMin + local_Nc -1 << endl;

  if(my_rank==MASTER){result.open("phi_alpha_mpi.datout");}
  // while(alpha<0.5)
  //   {
  //alpha += alphaIncr;
      if(my_rank==MASTER)
	{
	  cout << local_Nc << endl;
	  for(int i=0;i<Nc;i++){x0[i].x=0.5; x0[i].y=0.5; x0[i].z=0.5;}
	  cout << " " << endl;
	  MPI_Bcast(&x0[0], 3*Nc, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	}
      else
	{
	  MPI_Bcast(&x0[0], 3*Nc, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	}
      
      /*Transients*/
      for(int j=cloneIdxMin;j<local_Nc;j++)
	{
	  
	  //	  cout << "PROCESS " << my_rank << " | Clone " << j << " : " << x0[j].x << endl;

	  lorenz_transient(&x0[j], 5.0); 
	}
      /*Actual time evolution*/
      for(int t=0;t<nbrTimeSteps;t++)
	{
	  R = 0.0;
	  for(int j=cloneIdxMin;j<local_Nc;j++) // Loop on clones
	{
	  s[j] =  lorenz(x0[j], &x[j], alpha);
	  R+=s[j];
	}
	  if(my_rank==MASTER)
	    {
	      total_R = R;
	      for(int source=1;source<p;source++)
		{
		  tag = 0;
		  //MPI_Recv(&x[source*local_Nc], 3*local_Nc, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
		  MPI_Recv(&x[source*local_Nc].x, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
		  tag = 1;
		  MPI_Recv(&s[source*local_Nc], 3*local_Nc, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
		  tag = 3;
		  MPI_Recv(&R, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
		  total_R += R;
		}
	    }
	      else
		{
		  tag = 0;
		  //MPI_Send(&x[cloneIdxMin], 3*local_Nc, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
		  MPI_Send(&x[cloneIdxMin].x, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
		  
		  tag = 1;
		  MPI_Send(&s[cloneIdxMin], 3*local_Nc, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
		  tag = 3;
		  MPI_Send(&R, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
		}
	  MPI_Barrier(MPI_COMM_WORLD);	      
	      if(my_rank==MASTER)
		{
		  cout << "clone 5 " << x[5].x << endl;	  
	      total_R /= Nc;
	      cout << total_R << endl;
	      NcPrime = 0;
	      for(int j=0;j<Nc;j++)
		{
		  nbrCreatedCopies[j] = floor(s[j]/total_R + drand48());
		  NcPrime += nbrCreatedCopies[j]; 
		}
	      deltaN = NcPrime - Nc;
	      k=0;
	      for(int j=0;j<Nc;j++)
		{
		  for(int i=0;i<nbrCreatedCopies[j];i++)
		    {
		      temp[k] = x[j];
		      k++;
		    }
		}

	      if(deltaN > 0) // if too many copies created we prune
		{
		  for(int i=0;i<deltaN;i++)
		    {
		      idx = floor(NcPrime*drand48());
		      temp[idx] = temp[NcPrime-i];
		    }
		}
	      else if(deltaN < 0) // else we add just the right number of clones at random
		{
		  for(int i=0;i<-deltaN;i++)
		    {
		      idx = floor(NcPrime*drand48());
		      temp[NcPrime+i] = temp[idx];
		    }
		}

	      // CREATE INITIAL CONDITIONS FOR NEXT TIME EVOLUTION
	      k=0;
	      for(int j=0;j<Nc;j++)
		{
		  x0[j].x = randNormal(temp[j].x,eps);
		  x0[j].y = randNormal(temp[j].y,eps);
		  x0[j].z = randNormal(temp[j].z,eps);
		}
      
	      R_record[t] = total_R; // Store mean weight values for SCGF calculation later on
		} /*IF MASTER*/
	      MPI_Bcast(&x0[0], 3*Nc, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	} //END OF TIMESTEPS
      if(my_rank==MASTER)
	{
	  // COMPUTE SCGF
	  phi_alpha = 0;
	  for(int n=0;n<nbrTimeSteps;n++){phi_alpha += log(R_record[n]);}
	  phi_alpha /= T;
	  // WRITE RESULTING SCGF ON DISK
	  result << alpha << " " << phi_alpha << endl;
	}	  
      //}// END OF LOOP ON ALPHA


      MPI_Finalize();
}
      
double lorenz(Point xIn, Point *m_x, double alpha)
{
  /* Simulates the OU process during dT for 1 clone and compute the corresponding weight s defined as
     exp(\aplha * \int_{t}^{t+dT} x^{2}*dt) */
  double s = 0;
  Point xOut, temp;
  int nbrTimeStepsEvol = dT/dt;

  for(int t=0;t<nbrTimeStepsEvol;t++)
    {
      xOut.x = xIn.x + dt*sigma*(xIn.y-xIn.x);
      xOut.y = xIn.y + dt*(rho*xIn.x - xIn.y - xIn.x*xIn.z);
      xOut.z = xIn.z + dt*(xIn.x*xIn.y - beta*xIn.z);
      s += xOut.x/(5*sigma);
      temp = xIn;
      xIn = xOut;
      xOut = temp;
    }
  *m_x = xOut;
  s *= dt;
  //cout << s << endl;
  //cout << exp(alpha*s) << endl;
  return exp(alpha*s);
}
  
void lorenz_transient(Point *m_x0, double T_transient)
{

  /* Simulates the OU process during T_transient but does not compute any weight. It initializes the initial conditions array x0[] */
  Point xIn, xOut, temp;
  int nbrTimeStepsEvol = T_transient/dt;

  xIn = *m_x0;

  for(double t=0;t<nbrTimeStepsEvol;t++)
    {
      xOut.x = xIn.x + dt*sigma*(xIn.y-xIn.x);
      xOut.y = xIn.y + dt*(rho*xIn.x - xIn.y - xIn.x*xIn.z);
      xOut.z = xIn.z + dt*(xIn.x*xIn.y - beta*xIn.z);
      temp = xIn;
      xIn = xOut;
      xOut = temp;
    }
  *m_x0 = xOut; 
}

double randNormal(const double mean_, const double sigma_)
{
  /* Return a random number sampled in N(mean_, sigma_).
     Box-Muller method.
  */

  double x1, x2, w;
  do {
    x1 = 2.0 * (rand () / double (RAND_MAX)) - 1.0;
    x2 = 2.0 * (rand () / double (RAND_MAX)) - 1.0;
    w = x1 * x1 + x2 * x2;
  } while (w >= 1.0);

  w = sqrt (-2.0 * log (w)/w);
  const double y1 = x1 * w;
  const double y2 = x2 * w;

  return mean_ + y1 * sigma_;
}

