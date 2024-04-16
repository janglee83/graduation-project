export const getListEmployees = async () => {
    return await fetch('/demo/data/customers-large.json', { headers: { 'Cache-Control': 'no-cache' } })
        .then((response) => response.json())
        .then((data) => data.data);
}
