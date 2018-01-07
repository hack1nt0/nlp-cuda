
int main() {
	int a = 10;
#pragma omp parallel for
	for (int i = 0; i < 10; ++i);
	return 0;
}