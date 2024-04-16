<script setup lang="ts">
  import { ref, onBeforeMount } from 'vue';
  import Calendar from 'primevue/calendar';
  import Column from 'primevue/column';
  import InputText from 'primevue/inputtext';
  import InputIcon from 'primevue/inputicon';
  import IconField from 'primevue/iconfield';
  import DataTable from 'primevue/datatable';
  import Button from 'primevue/button';
  import { getListEmployees } from '../../../service/EmployeeService';

  interface ListEmployeesInterface {
    id: number;
    name: string;
    country: {
      name: string;
      code: string;
    };
    representative?: {
      name: string;
      image: string;
    };
    company: string;
    date: string;
    status: string;
    activity?: number;
    score?: number;
    balance: number;
  }
  const listEmployees = ref<Array<ListEmployeesInterface>>();
  const filtersValue = ref();
  const isLoading = ref<boolean>();

  onBeforeMount(() => {
    getListEmployees().then((data) => {
      listEmployees.value = data;
      listEmployees.value?.forEach((item) => {
        if (item.activity !== undefined) item.score = item.activity / 100;
        delete item.activity;
        delete item.representative;
      });
    });
  });

  const onClearFilter = () => {
    console.log('123');
  };

  const formatDate = (value: string): string => {
    const date = new Date(value);
    return date.toLocaleDateString('en-US', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    });
  };
</script>

<template>
  <div class="col-12">
    <div class="card">
      <h5>Thông tin nhân viên</h5>
      <DataTable
        :value="listEmployees"
        :paginator="true"
        :rows="10"
        dataKey="id"
        :rowHover="true"
        filterDisplay="menu"
        :globalFilterFields="['name', 'balance', 'status']"
        showGridlines
        :filters="filtersValue"
        :loading="isLoading"
      >
        <template #header>
          <div class="flex justify-content-between flex-column sm:flex-row">
            <Button
              type="button"
              icon="pi pi-filter-slash"
              label="Clear"
              outlined
              @click="onClearFilter()"
            />
            <IconField iconPosition="left">
              <InputIcon class="pi pi-search" />
              <!-- <InputText
                  v-model="filtersValue['global'].value"
                  placeholder="Keyword Search"
                  style="width: 100%"
                /> -->
            </IconField>
          </div>
        </template>
        <template #empty> No Employee found. </template>
        <template #loading> Loading Employee data. Please wait. </template>
        <Column field="name" header="Name" style="min-width: 12rem">
          <template #body="{ data }">
            {{ data.name }}
          </template>
          <template #filter="{ filterModel }">
            <InputText
              type="text"
              v-model="filterModel.value"
              class="p-column-filter"
              placeholder="Search by name"
            />
          </template>
        </Column>
        <Column
          header="Ngày vào công ty"
          filterField="date"
          dataType="date"
          style="min-width: 10rem"
        >
          <template #body="{ data }">
            {{ formatDate(data.date) }}
          </template>
          <template #filter="{ filterModel }">
            <Calendar v-model="filterModel.value" dateFormat="mm/dd/yy" placeholder="mm/dd/yyyy" />
          </template>
        </Column>
        <Column field="score" header="Score" :showFilterMatchModes="false" style="min-width: 12rem">
          <template #body="{ data }">
            {{ data.score }}
          </template>
        </Column>
      </DataTable>
    </div>
  </div>
</template>
