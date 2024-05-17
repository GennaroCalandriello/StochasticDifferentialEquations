/******************************************************************************

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java,
PHP, Ruby, Perl, C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C,
Assembly, HTML, CSS, JS, SQLite, Prolog. Code, Compile, Run and Debug online
from anywhere in world.

*******************************************************************************/
#include <iostream>

using namespace std;
long long int factorial(int n);
bool validadati(int n);
int main() {
  int n;
  int fatt = 1;
  cout << "Inserisci il numero\n";
  cin >> n;
  bool validate = validadati(n);
  while (validate == false) {
    cout << "Il numero deve essere positivo, reinserire\n";
    cin >> n;
  }
  long long int fact = factorial(n);

  cout << "Il fattoriale del numero inserito è " << fact;
  return 0;
}
bool validadati(int n) {
  if (n < 0) {
    return (false);
  } else {
    return (true);
  }
}

long long int factorial(int n) {

  if (n == 0)
    return 1;
  return n * factorial(n - 1);
};