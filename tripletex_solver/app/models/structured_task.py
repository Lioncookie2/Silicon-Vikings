from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

TaskType = Literal[
    # 1. Master Data
    "CREATE_EMPLOYEE", "CREATE_CUSTOMER", "CREATE_SUPPLIER", "CREATE_PRODUCT", "CREATE_PROJECT",
    # 2. Sales and Invoicing
    "CREATE_ORDER", "CREATE_INVOICE", "SEND_INVOICE", "REGISTER_INCOMING_PAYMENT",
    # 3. Expenses and Payables
    "REGISTER_SUPPLIER_INVOICE", "REGISTER_TRAVEL_EXPENSE",
    # 4. Ledger Correction
    "FIND_AND_FIX_VOUCHER_ERROR", "DELETE_DUPLICATE_VOUCHER", "ISSUE_REMINDER_FEE",
    # 5. HR and Time
    "REGISTER_EMPLOYMENT_AND_SALARY", "REGISTER_TIMESHEET", "CREATE_PAYROLL",
    # 6. Configuration
    "CREATE_ACCOUNTING_DIMENSION",
    "UNKNOWN",
    "COMPLEX_TASK"
]

class EmployeeData(BaseModel):
    firstName: str = Field(description="First name of the employee")
    lastName: str = Field(description="Last name of the employee")
    email: Optional[str] = Field(None, description="Email of the employee")
    isAccountAdministrator: Optional[bool] = Field(False)
    
class CustomerData(BaseModel):
    name: str = Field(description="Name of the customer company or person")
    email: Optional[str] = Field(None)
    organizationNumber: Optional[str] = Field(None)

class SupplierData(BaseModel):
    name: str = Field(description="Name of the supplier")
    organizationNumber: Optional[str] = Field(None)

class ProductData(BaseModel):
    name: str = Field(description="Product name")
    priceExVat: float = Field(description="Price excluding VAT")
    productNumber: Optional[str] = Field(None, description="Product number or ID")
    isVatFree: bool = Field(False, description="True if the product is VAT free or exempt (e.g. 0% VAT)")

class ProjectData(BaseModel):
    name: str = Field(description="Project name")
    customerName: str = Field(description="Customer to link project to")
    projectManagerName: Optional[str] = Field(None, description="Employee name of project manager")

class OrderData(BaseModel):
    customerName: str = Field(description="Customer to order for")
    productName: str = Field(description="Product name")
    quantity: int = Field(default=1)
    unitPrice: float = Field(description="Price per unit ex vat")
    orderDate: Optional[str] = Field(None, description="YYYY-MM-DD")

class InvoiceData(BaseModel):
    customerName: str = Field(description="Customer to invoice")
    invoiceDate: Optional[str] = Field(None)
    invoiceDueDate: Optional[str] = Field(None)
    amount: Optional[float] = Field(None)
    sendInvoice: bool = Field(False, description="Whether the invoice should also be sent")
    isCreditNote: bool = Field(False, description="True if this is a credit note or reversing an existing invoice")

class PaymentData(BaseModel):
    customerName: str = Field(description="Name of the customer who paid")
    amount: float = Field(description="The amount paid")
    organizationNumber: Optional[str] = Field(None)

class SupplierInvoiceData(BaseModel):
    supplierName: str = Field(description="Supplier who sent the invoice")
    amount: float = Field(description="Total amount")
    description: str = Field(description="Description of the expense (e.g. Rent, Equipment)")

class TravelExpenseData(BaseModel):
    employeeName: str = Field(description="Employee taking the trip")
    title: str = Field(description="Trip title")
    date: Optional[str] = Field(None, description="YYYY-MM-DD")

class VoucherCorrectionData(BaseModel):
    description: str = Field(description="What needs to be corrected")

class EmploymentAndSalaryData(BaseModel):
    employeeName: str = Field(description="Employee name")
    startDate: str = Field(description="YYYY-MM-DD")
    annualSalary: float = Field(description="Annual salary amount")

class StructuredTask(BaseModel):
    task_type: TaskType = Field(description="The primary type of accounting task identified")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    language: str = Field(description="Language code of the prompt (nb, en, es, pt, nn, de, fr)")
    
    # Generic catch-all for reasoning and other fields
    reasoning: str = Field(description="A short explanation of how the LLM arrived at this classification")
    extracted_entities: Optional[str] = Field(None, description="Stringified JSON of any other entities extracted from the prompt")

    # Specific Data Models
    employee_data: Optional[EmployeeData] = None
    customer_data: Optional[CustomerData] = None
    supplier_data: Optional[SupplierData] = None
    product_data: Optional[ProductData] = None
    project_data: Optional[ProjectData] = None
    order_data: Optional[OrderData] = None
    invoice_data: Optional[InvoiceData] = None
    payment_data: Optional[PaymentData] = None
    supplier_invoice_data: Optional[SupplierInvoiceData] = None
    travel_expense_data: Optional[TravelExpenseData] = None
    employment_salary_data: Optional[EmploymentAndSalaryData] = None
    voucher_correction_data: Optional[VoucherCorrectionData] = None
